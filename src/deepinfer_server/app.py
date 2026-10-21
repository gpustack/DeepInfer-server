import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path

from deepseek_recipe import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversionError,
    InferenceChunk,
    InferenceFinishReason,
    PromptUsage,
    StreamProcessor,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .backend import Backend
from .config import Settings
from .protocol import Protocol, ServiceError

logger = logging.getLogger(__name__)


def resolve_model(model: str) -> str:
    path = Path(model).expanduser()
    if not path.is_dir():
        if model.startswith(("/", "./", "../", "~")):
            raise ValueError(f"Model directory does not exist: {model}")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model))
    for filename in ("config.json", "tokenizer.json"):
        if not (path / filename).is_file():
            raise ValueError(f"Model directory requires {filename}: {path}")
    return str(path.resolve())


class InferenceResponse(StreamingResponse):
    def __init__(self, content, cancel):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        self.cancel = cancel

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cancel()
            await self.body_iterator.aclose()


def create_app(settings: Settings, *, backend=None, protocol=None) -> FastAPI:
    settings.validate()

    @asynccontextmanager
    async def lifespan(app):
        if backend is None:
            model_path = await run_in_threadpool(resolve_model, settings.model)
            runtime_settings = replace(settings, model=model_path)
            app.state.protocol = Protocol(
                runtime_settings, str(Path(model_path) / "tokenizer.json")
            )
            app.state.backend = Backend(runtime_settings)
        else:
            app.state.protocol = protocol
            app.state.backend = backend
        try:
            await app.state.backend.start()
            yield
        finally:
            await app.state.backend.close()

    app = FastAPI(title="DeepInfer Server", lifespan=lifespan)

    @app.exception_handler(ServiceError)
    async def service_error(_request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(ConversionError)
    async def conversion_error(_request, exc):
        return Response(exc.body, status_code=exc.status_code, media_type="application/json")

    @app.get("/health")
    async def health():
        healthy = app.state.backend.healthy
        return JSONResponse(
            {"status": "ok" if healthy else "unavailable"}, status_code=200 if healthy else 503
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "deepinfer",
                }
            ],
        }

    async def protocol_chunks(prepared, handle):
        processor = None
        completion_tokens = 0
        engine = app.state.backend
        try:
            generator = ChatCompletionRequest.chunk_generator(
                prepared.converted,
                handle.request_id,
                prepared.model,
            )
            processor = StreamProcessor(
                generator,
                prepared.converted.parsing_options,
                app.state.protocol.tokenizer,
            )
            async with asyncio.timeout(settings.request_timeout):
                ready = InferenceChunk.ready(
                    prompt_usage=PromptUsage(
                        prompt_tokens=len(prepared.token_ids),
                    )
                )
                for chunk in processor.push(ready):
                    yield chunk, completion_tokens
                done = False
                while not done and not processor.finished:
                    updates = await handle.queue.get()
                    if isinstance(updates, Exception):
                        raise updates
                    for update in updates:
                        if "error" in update:
                            raise ServiceError(update["error"], 400)
                        reason = update["finish_reason"]
                        if update["finished"] and reason not in ("stop", "length"):
                            raise ServiceError(f"Inference ended with {reason}", 503)
                        # EOS counts toward usage but is not visible response text.
                        completion_tokens += 1
                        token_chunk = (
                            InferenceChunk.text("", token_count=1)
                            if update["token_id"] == app.state.protocol.eos_token_id
                            else InferenceChunk.tokens([update["token_id"]])
                        )
                        for chunk in processor.push(token_chunk):
                            yield chunk, completion_tokens
                        if processor.finished:
                            break
                        if update["finished"]:
                            finish = (
                                InferenceFinishReason.Length
                                if reason == "length"
                                else InferenceFinishReason.Stop
                            )
                            for chunk in processor.push(InferenceChunk.finish(finish)):
                                yield chunk, completion_tokens
                            done = True
                            break
                for chunk in processor.finish():
                    yield chunk, completion_tokens
        except TimeoutError as exc:
            raise ServiceError("Inference request timed out", 504) from exc
        finally:
            engine.cancel(handle)
            if processor is not None:
                processor.close()

    async def stream_body(prepared, handle):
        chunks = protocol_chunks(prepared, handle)
        usage_chunk = None
        try:
            async for chunk, completion_tokens in chunks:
                data = json.loads(chunk.to_json())
                if data.get("usage") is not None:
                    usage_chunk = {**data, "choices": []}
                if prepared.include_usage and prepared.continuous_usage_stats:
                    # Count generated tokens, including EOS and tokens buffered by the parser.
                    data["usage"] = data.get("usage") or {
                        "prompt_tokens": len(prepared.token_ids),
                        "completion_tokens": completion_tokens,
                        "total_tokens": len(prepared.token_ids) + completion_tokens,
                    }
                else:
                    data["usage"] = None
                yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
            if prepared.include_usage and usage_chunk is not None:
                yield "data: " + json.dumps(usage_chunk, ensure_ascii=False) + "\n\n"
        except ServiceError as exc:
            yield "data: " + json.dumps(exc.body()) + "\n\n"
        except Exception:
            logger.exception("Streaming response failed")
            yield "data: " + json.dumps(ServiceError("Inference failed", 500).body()) + "\n\n"
        finally:
            await chunks.aclose()
        yield "data: [DONE]\n\n"

    async def collect(prepared, handle):
        response = ChatCompletionResponse(handle.request_id, prepared.model, int(time.time()), 0, 0)
        chunks = protocol_chunks(prepared, handle)
        try:
            async for chunk, _ in chunks:
                response.accumulate(chunk)
            return Response(response.to_json(), media_type="application/json")
        except (ServiceError, ConversionError):
            raise
        except Exception as exc:
            logger.exception("Response processing failed")
            raise ServiceError("Inference failed", 500) from exc
        finally:
            await chunks.aclose()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        raw = bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw) > 2 * 1024 * 1024:
                raise ServiceError("Request body exceeds 2 MiB", 413)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ServiceError("Invalid JSON request body") from exc
        try:
            prepared = await run_in_threadpool(app.state.protocol.prepare, body)
        except (ValueError, TypeError) as exc:
            raise ServiceError(str(exc)) from exc
        handle = app.state.backend.submit(prepared)
        if prepared.stream:
            return InferenceResponse(
                stream_body(prepared, handle), lambda: app.state.backend.cancel(handle)
            )

        disconnect_stop = asyncio.Event()

        async def disconnected():
            while not disconnect_stop.is_set() and not await request.is_disconnected():
                await asyncio.sleep(0.2)

        response_task = asyncio.create_task(collect(prepared, handle))
        disconnect_task = asyncio.create_task(disconnected())
        try:
            await asyncio.wait(
                (response_task, disconnect_task), return_when=asyncio.FIRST_COMPLETED
            )
            if response_task.done():
                return response_task.result()
            raise ServiceError("Client disconnected", 499)
        finally:
            disconnect_stop.set()
            for task in (response_task, disconnect_task):
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            app.state.backend.cancel(handle)

    return app

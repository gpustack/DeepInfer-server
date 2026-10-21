import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from deepseek_recipe import SpecialToken
from fastapi.testclient import TestClient
from openai import OpenAI

from deepinfer_server.app import InferenceResponse, create_app
from deepinfer_server.protocol import ServiceError

from .fakes import FakeBackend


def test_sdk_completion(settings, protocol):
    backend = FakeBackend(protocol.tokenizer)
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)
        result = sdk.chat.completions.create(
            model=settings.served_model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
        )
        assert result.choices[0].message.content == "Hello, world! 🌍"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert backend.submitted[0].sampling["max_tokens"] == 128
        assert client.get("/health").status_code == 200
        assert sdk.models.list().data[0].id == settings.served_model_name
    assert backend.cancelled


@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("continuous_usage_stats", [None, False, True])
def test_sdk_stream(settings, protocol, include_usage, continuous_usage_stats):
    backend = FakeBackend(protocol.tokenizer)
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)
        chunks = list(
            sdk.chat.completions.create(
                model=settings.served_model_name,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_options={"include_usage": include_usage},
                extra_body={
                    "thinking": {"type": "disabled"},
                    **(
                        {
                            "stream_options": {
                                "include_usage": include_usage,
                                "continuous_usage_stats": continuous_usage_stats,
                            }
                        }
                        if continuous_usage_stats is not None
                        else {}
                    ),
                },
            )
        )
        assert (
            "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
            == "Hello, world! 🌍"
        )
        assert any(c.choices and c.choices[0].finish_reason == "stop" for c in chunks)
        if include_usage:
            assert chunks[-1].choices == []
            assert chunks[-1].usage.total_tokens > 0
            expected_tokens = len(protocol.tokenizer.encode(backend.text))
            assert chunks[-1].usage.completion_tokens == expected_tokens
            if continuous_usage_stats:
                counts = [c.usage.completion_tokens for c in chunks]
                assert counts[0] == 0
                assert counts == sorted(counts)
                assert any(0 < n < expected_tokens for n in counts)
                for chunk in chunks:
                    usage = chunk.usage
                    assert usage.prompt_tokens == len(backend.submitted[0].token_ids)
                    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
            else:
                assert all(c.usage is None for c in chunks[:-1])
        else:
            assert all(c.usage is None for c in chunks)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_eos_is_counted_without_visible_text(settings, protocol, body, stream, ignore_eos):
    suffix = " More text." if ignore_eos else ""
    text = "Hello!" + SpecialToken.Eos.default_str() + suffix
    reason = "length" if ignore_eos else "stop"
    backend = FakeBackend(protocol.tokenizer, text, reason=reason)
    body["ignore_eos"] = ignore_eos
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        if stream:
            chunks = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            content = "".join(
                choice["delta"].get("content") or ""
                for chunk in chunks
                for choice in chunk["choices"]
            )
            usage = chunks[-1]["usage"]
            counts = [chunk["usage"]["completion_tokens"] for chunk in chunks]
            assert counts == sorted(counts)
            assert all(chunk["usage"] is not None for chunk in chunks)
            assert any(
                choice["finish_reason"] == reason for chunk in chunks for choice in chunk["choices"]
            )
        else:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            usage = result["usage"]
            assert result["choices"][0]["finish_reason"] == reason
        assert backend.submitted[0].sampling["ignore_eos"] is ignore_eos
        assert content == "Hello!" + suffix
        assert usage["completion_tokens"] == len(protocol.tokenizer.encode(text))


def test_reasoning(settings, protocol, body):
    backend = FakeBackend(protocol.tokenizer, "Let me think.</think>The answer is four.")
    body["thinking"] = {"type": "enabled"}
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        result = client.post("/v1/chat/completions", json=body)
        assert result.status_code == 200, result.text
        message = result.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "Let me think."
        assert message["content"] == "The answer is four."


def test_tool_call_parsing(settings, protocol, body):
    text = (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="weather">\n'
        '<｜DSML｜parameter name="city" string="true">London</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
    backend = FakeBackend(protocol.tokenizer, text)
    body["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                },
            },
        }
    ]
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        result = client.post("/v1/chat/completions", json=body)
        assert result.status_code == 200, result.text
        choice = result.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        function = choice["message"]["tool_calls"][0]["function"]
        assert function["name"] == "weather"
        assert json.loads(function["arguments"]) == {"city": "London"}


def test_stop_sequence_cancels(settings, protocol, body):
    backend = FakeBackend(protocol.tokenizer, "hello STOP hidden text")
    body["stop"] = ["STOP"]
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        result = client.post("/v1/chat/completions", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["choices"][0]["message"]["content"] == "hello "
        assert backend.cancelled


def test_length_finish(settings, protocol, body):
    backend = FakeBackend(protocol.tokenizer, reason="length")
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        result = client.post("/v1/chat/completions", json=body)
        assert result.json()["choices"][0]["finish_reason"] == "length"


def test_health_and_model_listing(settings, protocol):
    backend = FakeBackend(protocol.tokenizer)
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        backend.healthy = False
        assert client.get("/health").status_code == 503


@pytest.mark.parametrize("stream", [False, True])
def test_timeout_releases_request(settings, protocol, body, stream):
    backend = FakeBackend(protocol.tokenizer, stall=True)
    settings = replace(settings, request_timeout=0.03)
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        response = client.post("/v1/chat/completions", json={**body, "stream": stream})
        if stream:
            assert response.status_code == 200
            assert '"error"' in response.text and "timed out" in response.text
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.status_code == 504
        assert backend.cancelled


@pytest.mark.parametrize("stream", [False, True])
def test_backend_failure(settings, protocol, body, stream):
    backend = FakeBackend(protocol.tokenizer, error=ServiceError("Engine unavailable", 503))
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        result = client.post("/v1/chat/completions", json={**body, "stream": stream})
        assert result.status_code == (200 if stream else 503)
        assert "Engine unavailable" in result.text
        assert backend.cancelled


@pytest.mark.parametrize(
    "patch",
    [
        {"n": 2},
        {"response_format": {"type": "json_object"}},
        {"tool_choice": "required"},
        {"logprobs": True},
        {"frequency_penalty": 1},
        {"messages": []},
        {"stream": "true"},
        {"stream_options": []},
        {"stream_options": {"include_usage": "true"}},
        {"stream_options": {"continuous_usage_stats": "true"}},
        {"stream_options": {"continuous_usage_stats": 1}},
        {"stream_options": {"continuous_usage_stats": None}},
        {"stream_options": {"unknown_option": True}},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://example.com"}}],
                }
            ]
        },
        {"max_tokens": 5000},
        {"top_p": -1},
        {"arbitrary": 1},
        {"ignore_eos": "true"},
        {"ignore_eos": 1},
        {"ignore_eos": None},
    ],
)
def test_invalid_requests_never_reach_engine(settings, protocol, body, patch):
    backend = FakeBackend(protocol.tokenizer)
    with TestClient(create_app(settings, backend=backend, protocol=protocol)) as client:
        result = client.post("/v1/chat/completions", json={**body, **patch})
        assert result.status_code == 400, result.text
        assert "error" in result.json()
        assert not backend.submitted


async def test_stream_disconnect_closes_generator():
    closed = asyncio.Event()
    cancelled = []

    async def content():
        try:
            yield "data: first\n\n"
            await asyncio.sleep(60)
        finally:
            closed.set()

    sent = asyncio.Event()

    async def receive():
        await sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            sent.set()

    response = InferenceResponse(content(), lambda: cancelled.append(True))
    await asyncio.wait_for(
        response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send), 2
    )
    assert closed.is_set() and cancelled


async def test_nonstream_disconnect_cancels(settings, protocol, body):
    backend = FakeBackend(protocol.tokenizer, stall=True)
    app = create_app(settings, backend=backend, protocol=protocol)
    raw = json.dumps(body).encode()
    first = True

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": raw, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 1234),
    }
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), 3)
        assert backend.cancelled


async def test_concurrent_requests_are_isolated(settings, protocol, body):
    backend = FakeBackend(protocol.tokenizer)
    app = create_app(settings, backend=backend, protocol=protocol)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            responses = await asyncio.gather(
                *[c.post("/v1/chat/completions", json=body) for _ in range(5)]
            )
    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["id"] for r in responses}) == 5

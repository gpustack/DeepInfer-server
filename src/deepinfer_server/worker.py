"""One process owns each DeepInfer driver and its CP workers."""

import os
import traceback

from .config import Settings


class Engine:
    def __init__(self, settings: Settings, rank: int, devices: list[str]):
        os.environ["DEEPINFER_ACCELERATOR"] = "nvidia"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("DEEPINFER_USE_NEXTN_DECODE", "0")
        os.environ.setdefault("DEEPINFER_SAMPLER_FORCE_TOPK", "4096")
        os.environ.setdefault("DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS", "32")
        os.environ.setdefault("DEEPINFER_EXPERT_COUNT_WINDOW_SIZE", "3")
        # Load PyTorch's shared libraries after device assignment and before native extensions.
        import numpy as np
        import torch

        available = torch.cuda.device_count()
        if available != len(devices):
            raise ValueError(
                f"Worker {rank} requires {len(devices)} CUDA devices, but only {available} are "
                "visible. Configure --expert-parallel-size, --dp, and --cp to match the GPUs."
            )

        from deepinfer._C.engine import RequestOutput
        from deepinfer.engine.arg_utils import EngineArgs
        from deepinfer.engine.llm_engine import LLMEngine

        self.np = np
        self.output_type = RequestOutput
        self.engine = LLMEngine.from_engine_args(
            EngineArgs(
                model=settings.model,
                served_model_name=settings.served_model_name,
                trust_remote_code=settings.trust_remote_code,
                expert_parallel_size=settings.expert_parallel_size,
                data_parallel_size=settings.data_parallel_size,
                context_parallel_size=settings.context_parallel_size,
                engine_rank=rank,
                max_model_len=settings.max_model_len,
                max_num_seqs=settings.max_num_seqs,
                max_num_batched_tokens=settings.max_num_batched_tokens,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                chunked_prefill=True,
                stop_preempted_request=False,
                lookahead_tokens=min(1, int(os.environ["DEEPINFER_USE_NEXTN_DECODE"])),
            )
        )

    def add(self, request_id: str, token_ids: list[int], sampling: dict) -> None:
        self.engine.add_request(
            user_id="api",
            request_id=request_id,
            sampling_params=sampling,
            prompt_token_ids=self.np.asarray(token_ids, dtype=self.np.int32),
            multi_modal_data={"image_bytes": [], "image_sizes": []},
        )

    def abort(self, request_id: str) -> None:
        self.engine.abort_request(request_id)

    def tick(self) -> tuple[bool, list[dict]]:
        # Every DP rank enters this collective once in each coordinator round.
        active = self.engine.has_unfinished_requests()
        raw = self.engine.step() if active else self.engine.flush_inflight_reqs()
        outputs = []
        for pointer in raw:
            output = self.output_type.clone(pointer)
            # Copy values before the next step releases DeepInfer's output pool.
            outputs.append(
                {
                    "request_id": output.request_id,
                    "token_id": output.token_id,
                    "finished": output.finished,
                    "finish_reason": output.finish_reason,
                }
            )
        return bool(active), outputs

    def close(self) -> None:
        self.engine.close()


def run_worker(
    connection, settings: Settings, rank: int, devices: list[str], factory, master_port: str
):
    os.setsid()
    os.environ["MASTER_PORT"] = master_port
    engine = None
    try:
        engine = factory(settings, rank, devices)
        connection.send(("ready", None))
        while True:
            kind, commands = connection.recv()
            if kind == "stop":
                break
            if kind != "tick":
                raise RuntimeError(f"Unknown coordinator command: {kind}")
            errors = []
            for operation, request_id, payload in commands:
                if operation == "add":
                    try:
                        engine.add(request_id, *payload)
                    except ValueError as exc:
                        errors.append({"request_id": request_id, "error": str(exc)})
                elif operation == "cancel":
                    engine.abort(request_id)
            active, outputs = engine.tick()
            connection.send(("result", (active, errors + outputs)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        traceback.print_exc()
        try:
            connection.send(("fatal", "DeepInfer worker failed; check server logs"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if engine is not None:
            engine.close()
        connection.close()

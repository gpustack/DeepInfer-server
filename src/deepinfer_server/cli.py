import argparse

from .config import Settings


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        prog="deepinfer-server", description="Serve a DeepSeek-V4.1-Flash checkpoint"
    )
    parser.add_argument("model", help="Local HF checkpoint directory or Hugging Face model ID")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="deepseek-flash")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--expert-parallel-size", "--ep", type=int, default=8)
    parser.add_argument("--data-parallel-size", "--dp", "-dp", type=int, default=1)
    parser.add_argument("--context-parallel-size", "--cp", type=int)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = vars(parser.parse_args(argv))
    if args.pop("tensor_parallel_size") != 1:
        parser.error("DeepInfer does not provide tensor parallelism; configure --ep/--dp/--cp")
    if args["context_parallel_size"] is None:
        dp = args["data_parallel_size"]
        if dp <= 0 or args["expert_parallel_size"] % dp:
            parser.error("--expert-parallel-size must be divisible by --data-parallel-size")
        args["context_parallel_size"] = args["expert_parallel_size"] // dp
    settings = Settings(**args)
    try:
        settings.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return settings


def main() -> None:
    settings = parse_args()
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)

# DeepInfer Server

A Chat Completions service built with DeepInfer and deepseek-recipe. Supports JSON and SSE responses, reasoning content, automatic tool-call parsing, `/v1/models`, and `/health`.

This minimal service runs DeepSeek-V4.1-Flash on a single node with multiple GPUs. It requires Linux, NVIDIA SM100 GPUs, and enough GPU memory for the model and KV cache.

## Build

Run on a Linux NVIDIA host:

```bash
docker build --build-arg MAX_JOBS=8 -t deepinfer-server:latest .
```

The image uses CUDA 13.2.0, Python 3.12, and deepseek-recipe 0.2.0. DeepInfer is pinned to commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`. Building downloads and compiles DeepInfer and its kernel dependencies, requiring network access, substantial disk space, and memory. CUDA build tools remain available at runtime for kernel JIT compilation. The dependency check allows DeepInfer's intentional NCCL 2.32.3 override of PyTorch's 2.29.7 pin; other dependency conflicts fail the build.

If FlashMLA finishes compilation but fails with `Register spilling detected`, its post-build performance check found kernels using stack or local memory. To allow those kernels, use the upstream override:

```bash
docker build --build-arg MAX_JOBS=8 \
  --build-arg FLASH_MLA_SKIP_REG_SPILL_CHECK=1 \
  -t deepinfer-server:latest .
```

This skips only FlashMLA's register-spill check; it does not eliminate spills or suppress compiler errors. Kernel performance may be lower. The check remains enabled by default.

## Run

This example uses eight visible GPUs with EP=8, DP=1, and CP=8. This is an example configuration, not a verified minimum GPU memory requirement.

```bash
docker run --rm --gpus all --device=/dev/infiniband --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v /path/to/DeepSeek-V4.1-Flash-1017:/models/DeepSeek-V4.1-Flash:ro \
  -v deepinfer-cache:/cache \
  deepinfer-server:latest \
  /models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-flash \
  --trust-remote-code \
  --expert-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --startup-timeout 3600
```

Mount a complete Hugging Face checkpoint containing the configuration, tokenizer, and safetensors weights. Alternatively, replace the model path with `deepseek-ai/DeepSeek-V4.1-Flash-1017` to download it into the cache on first startup.

The named Docker volume `deepinfer-cache` persists `/cache` across container replacement, including JIT artifacts in `/cache/jit` and Hugging Face downloads and remote-code modules in `/cache/huggingface`. It does not persist the GPU KV cache or eliminate model loading at startup.

The host needs an NVIDIA driver and NVIDIA Container Toolkit. GPU interconnects must meet DeepInfer/DeepEP requirements. This checkpoint uses RDMA for its CPU-resident Engram tables, so the example exposes `/dev/infiniband`. Configure the host fabric and any required NCCL network environment variables before starting the service; the image does not configure host interconnects.

The HTTP server starts listening after model initialization completes. Each service selects an available distributed initialization port and shares it across its DP/CP ranks. Set `MASTER_PORT` explicitly if needed; DeepInfer uses `MASTER_PORT - 50` as the actual rendezvous port, which must be free and different from the HTTP port.

Speculative decoding is disabled by default to simplify initial deployment. Set the relevant DeepInfer environment variables to enable it. Use `--dp` and `--cp` to configure parallelism, with `DP * CP = EP`. If CP is omitted, it is derived from EP/DP. The service runs one HTTP worker and manages all inference processes internally.

With an existing compiled DeepInfer environment, run the service directly:

```bash
python -m pip install .
export PYTHONPATH=/path/to/DeepInfer
export DEEPINFER_ACCELERATOR=nvidia
deepinfer-server /path/to/checkpoint --trust-remote-code --ep 8
```

Run `deepinfer-server --help` to see the available options.

## GPUStack Integration

[gpustack-deepinfer-server-backend.yaml](gpustack-deepinfer-server-backend.yaml) defines a custom GPUStack backend using `quay.io/enjoydemo/deepinfer-server:latest`. Import the file as a single YAML object when creating a custom inference backend. Its backend name is `DeepInfer-server-custom`, including the required `-custom` suffix.

Deploy a complete DeepSeek-V4.1-Flash checkpoint with the **DeepInfer-server-custom** backend and select the required GPUs on one worker. GPUStack supplies the model path, serving port, model name, and GPU count. The command uses EP equal to that count, DP=1, and the corresponding CP. For the eight-GPU example, select all eight GPUs on the same worker. The definition includes a one-hour startup timeout for cold weight loading and JIT compilation.

The definition includes the [DeepSeek organization logo](https://github.com/deepseek-ai). The upstream DeepInfer repository does not provide a separate logo. The referenced GPUStack custom YAML import route currently filters out `icon`, so displaying it requires GPUStack support for importing that field.

The worker must expose its RDMA devices to the inference container. Keep `GPUSTACK_HOST_IPC=true` in the model's environment settings. The entrypoint sets unlimited locked memory and a 64 MiB stack limit. The definition includes `NCCL_IB_HCA`, `NCCL_IB_SL`, and `EP_OVERRIDE_RDMA_SL` values validated on the eight-B200 worker. Override them in the model's environment to match other RDMA fabrics. The backend definition sets `DEEPINFER_JIT_CACHE_DIR=/var/lib/gpustack/cache/deepinfer/jit`. The inference container must have a writable persistent mount covering this path. GPUStack mirrored deployments can inherit the worker's cache mount; otherwise configure a persistent mount explicitly. Override this path if the worker uses a different data directory. Checkpoints downloaded by GPUStack remain in its model cache and are passed through `{{model_path}}`. Existing deployments may need the cache and RDMA environment values added explicitly.

If Engram initialization fails with `GIN: DevComm setup failed on all available backends`, check RDMA device access and the NCCL fabric configuration. The eight-B200 validation worker required the following model environment values:

```yaml
NCCL_IB_HCA: "=mlx5_0"
NCCL_IB_SL: "0"
EP_OVERRIDE_RDMA_SL: "0"
```

The leading `=` selects the exact HCA name. This combination passed the DeepEP Engram test and both Chat Completions modes on that worker. These values are included in the backend YAML. Use the HCA and service level appropriate for your own fabric.

## API usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "thinking": {"type": "disabled"},
    "max_tokens": 128,
    "stream": false
  }'
```

Set `stream` to `true` and add `-N` to curl to stream the response. Include `"stream_options":{"include_usage":true}` to receive token usage at the end of the stream. Add `"continuous_usage_stats":true` inside `stream_options` to also receive cumulative token usage in each chunk, as requested by benchmark clients such as GuideLLM.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
    extra_body={"thinking": {"type": "disabled"}},
)
print(response.choices[0].message.content)
```

For fixed-length benchmarks, send `"ignore_eos": true` with an explicit `max_tokens`. This forwards DeepInfer's native EOS override; the default is `false`. Output limits and explicit stop sequences still apply.

Thinking mode follows recipe defaults, and `max_tokens` includes reasoning tokens. Disable thinking when checking your initial deployment.

```bash
python scripts/smoke.py --base-url http://localhost:8000
```

The service supports text, `n=1`, and automatic or disabled tool calls. Images, logprobs, forced tool selection, and structured output constraints return errors. Clients execute tools; the service only generates and parses tool calls. Responses/Messages APIs, cross-request prefix caching, and disaggregated prefill/decode deployment are not provided.

The coordinator advances DP ranks in rounds to keep execution synchronized. This adds CPU and interprocess communication overhead; reproducing official offline benchmark throughput is not a goal. Requests exceeding capacity receive HTTP 429. Timeouts and client disconnects cancel inference. A worker failure makes the health endpoint return HTTP 503 and requires restarting the service.

## Local tests

```bash
python -m pip install '.[test]'
pytest
```

Tests use the real deepseek-recipe library, a test tokenizer, and simulated inference engines to exercise API behavior and process coordination. They require no GPUs and do not validate CUDA kernels, model loading, or real GPU collectives. Validate deployment by building the image and running the smoke test on the target GPU host.

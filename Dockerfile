ARG CUDA_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEEPINFER_REF=9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
ARG MAX_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive \
    DEEPINFER_ACCELERATOR=nvidia \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/DeepInfer \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=10.0a \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/cache/huggingface \
    DEEPINFER_JIT_CACHE_DIR=/cache/jit

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential gcc-13 g++-13 cmake ninja-build \
    python3.12 python3.12-dev python3.12-venv pkg-config pciutils \
    libnuma-dev numactl libibverbs-dev librdmacm-dev infiniband-diags libdw-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-13 CXX=g++-13
RUN python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel packaging

WORKDIR /opt/DeepInfer
ARG FLASH_MLA_SKIP_REG_SPILL_CHECK=0
RUN git init . \
    && git remote add origin https://github.com/deepseek-ai/DeepInfer.git \
    && git fetch --depth 1 origin "${DEEPINFER_REF}" \
    && git checkout --detach FETCH_HEAD \
    && FLASH_MLA_SKIP_REG_SPILL_CHECK="${FLASH_MLA_SKIP_REG_SPILL_CHECK}" ./build.sh

WORKDIR /opt/deepinfer-server
COPY pyproject.toml ./
COPY src ./src
COPY scripts/check_dependencies.py ./scripts/check_dependencies.py
RUN pip install --no-cache-dir . && python scripts/check_dependencies.py
COPY scripts/smoke.py ./scripts/smoke.py
EXPOSE 8000
STOPSIGNAL SIGTERM
ENTRYPOINT ["deepinfer-server"]
CMD ["/models/DeepSeek-V4.1-Flash", "--trust-remote-code"]

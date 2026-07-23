# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.11.15

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}@sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c AS llama-builder
ARG LLAMA_CPP_COMMIT=3ce7da2c852c538c4c5f9806da27029cf8c9cc4a
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" \
    && git checkout --detach "${LLAMA_CPP_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}" \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        "-DCMAKE_CUDA_ARCHITECTURES=86;89" \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --target llama-server --parallel "$(nproc)" \
    && strip build/bin/llama-server

FROM ubuntu:${UBUNTU_VERSION}@sha256:52df9b1ee71626e0088f7d400d5c6b5f7bb916f8f0c82b474289a4ece6cf3faf AS python-builder
ARG PYTHON_VERSION
ARG PYTHON_SOURCE_SHA256=f4de1b10bd6c70cbb9fa1cd71fc5038b832747a74ee59d599c69ce4846defb50
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl libbz2-dev libffi-dev liblzma-dev \
        libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev \
        uuid-dev xz-utils zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /tmp/Python.tgz \
        "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
    && echo "${PYTHON_SOURCE_SHA256}  /tmp/Python.tgz" | sha256sum --check \
    && tar -xzf /tmp/Python.tgz -C /tmp \
    && cd "/tmp/Python-${PYTHON_VERSION}" \
    && ./configure --prefix=/opt/python --enable-shared --with-ensurepip=install \
    && make --jobs "$(nproc)" \
    && make install \
    && LD_LIBRARY_PATH=/opt/python/lib /opt/python/bin/python3.11 --version \
    && rm -rf /tmp/Python.tgz "/tmp/Python-${PYTHON_VERSION}"
RUN rm -rf \
    /opt/python/lib/python3.11/test \
    /opt/python/lib/python3.11/ctypes/test \
    /opt/python/lib/python3.11/distutils/tests \
    /opt/python/lib/python3.11/email/test \
    /opt/python/lib/python3.11/lib2to3/tests \
    /opt/python/lib/python3.11/sqlite3/test \
    /opt/python/lib/python3.11/tkinter/test \
    /opt/python/lib/python3.11/unittest/test
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:733b4042187702f832f7fdecb3aff14a61b288c4ca37af188bb5715c1caebaf8 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/python/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/opt/python/lib
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md THIRD_PARTY.md ./
COPY packages/hugging-voice-protocol ./packages/hugging-voice-protocol
COPY services/gpu-service ./services/gpu-service
RUN uv sync --package hugging-voice-gpu-service --extra gpu --frozen --no-dev --no-editable

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9 AS runtime
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libbz2-1.0 libexpat1 libffi8 libgfortran5 libgomp1 liblzma5 \
        libncursesw6 libopenblas0 libreadline8t64 libsndfile1 libsqlite3-0 \
        libssl3 libstdc++6 libuuid1 zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 hugging-voice \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin hugging-voice \
    && install -d -o 10001 -g 10001 /cache /tmp/hugging-voice
COPY --from=python-builder /opt/python /opt/python
COPY --from=python-builder /opt/venv /opt/venv
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY services/gpu-service/config/default.yaml /etc/hugging-voice/config.yaml
COPY models/manifest.yaml /etc/hugging-voice/manifest.yaml
ARG LLAMA_CPP_COMMIT=3ce7da2c852c538c4c5f9806da27029cf8c9cc4a
ARG VCS_REF=unknown
ARG SOURCE_URL=unknown
LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.hugging-voice.llama-cpp-commit="${LLAMA_CPP_COMMIT}"
ENV PATH=/opt/venv/bin:/opt/python/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/opt/python/lib \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    TMPDIR=/tmp/hugging-voice \
    HV_MODELS__ROOT=/models \
    HV_MODELS__LOCK_FILE=/models/manifest.lock.json \
    HV_MODELS__LLAMA_SERVER_BINARY=/usr/local/bin/llama-server \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=15s --timeout=3s --start-period=15m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health/live', timeout=2).read()"]
ENTRYPOINT ["/opt/venv/bin/hugging-voice-service"]
CMD ["--config", "/etc/hugging-voice/config.yaml"]

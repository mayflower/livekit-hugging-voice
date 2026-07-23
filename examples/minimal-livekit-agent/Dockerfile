# syntax=docker/dockerfile:1.7
FROM python:3.14.6-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:733b4042187702f832f7fdecb3aff14a61b288c4ca37af188bb5715c1caebaf8 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md ./
COPY packages ./packages
COPY examples ./examples
RUN uv sync --package hugging-voice-minimal-agent --frozen --no-dev --no-editable

FROM python:3.14.6-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30
RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin agent
COPY --from=builder /opt/venv /opt/venv
COPY examples/minimal-livekit-agent/agent.py /app/agent.py
COPY examples/minimal-livekit-agent/web.py /app/web.py
COPY examples/minimal-livekit-agent/static /app/static
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin PYTHONUNBUFFERED=1
USER 10001:10001
ENTRYPOINT ["python", "/app/agent.py"]
CMD ["start"]

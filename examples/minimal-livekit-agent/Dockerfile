# syntax=docker/dockerfile:1.7
FROM python:3.11.13-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md ./
COPY packages ./packages
COPY examples ./examples
RUN uv sync --package hugging-voice-minimal-agent --frozen --no-dev --no-editable

FROM python:3.11.13-slim-bookworm
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

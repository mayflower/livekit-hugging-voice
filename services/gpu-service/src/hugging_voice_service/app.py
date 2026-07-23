"""FastAPI application exposing real lifecycle health and model metadata."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, WebSocket
from fastapi.responses import JSONResponse, Response

from .config import ServiceSettings, load_settings
from .lifecycle import ServiceLifecycle
from .realtime import RealtimeService


def create_app(
    settings: ServiceSettings,
    *,
    lifecycle: ServiceLifecycle | None = None,
    realtime: RealtimeService | None = None,
) -> FastAPI:
    service_lifecycle = lifecycle or ServiceLifecycle(settings)
    realtime_service = realtime or RealtimeService(service_lifecycle)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app

        async def start() -> None:
            await service_lifecycle.start()
            await realtime_service.start()

        startup_task = asyncio.create_task(start())
        try:
            yield
        finally:
            if not startup_task.done():
                startup_task.cancel()
            await asyncio.gather(startup_task, return_exceptions=True)
            await realtime_service.aclose()
            await service_lifecycle.aclose()

    app = FastAPI(title="Hugging Voice GPU Service", version="0.2.0", lifespan=lifespan)
    app.state.lifecycle = service_lifecycle
    app.state.realtime = realtime_service

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        authenticator = service_lifecycle.authenticator
        if authenticator is None:
            raise HTTPException(status_code=503, detail="service is not ready")
        if not authenticator.authenticate_header(authorization):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.get("/health/live")
    async def live() -> JSONResponse:
        status = 200 if service_lifecycle.live else 500
        return JSONResponse(
            status_code=status,
            content={
                "status": "live" if status == 200 else "failed",
                "phase": service_lifecycle.phase,
            },
        )

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        status = 200 if realtime_service.ready else 503
        return JSONResponse(
            status_code=status,
            content={
                "status": "ready" if status == 200 else "not_ready",
                "phase": service_lifecycle.phase,
            },
        )

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> dict[str, object]:
        await require_auth(authorization)
        return service_lifecycle.model_report()

    @app.get("/v1/capacity")
    async def capacity(authorization: str | None = Header(default=None)) -> dict[str, int]:
        await require_auth(authorization)
        return await realtime_service.capacity.report()

    @app.get("/v1/pool")
    async def pool(authorization: str | None = Header(default=None)) -> list[dict[str, object]]:
        await require_auth(authorization)
        return await realtime_service.capacity.pool_report()

    @app.get("/v1/usage")
    async def usage(authorization: str | None = Header(default=None)) -> dict[str, object]:
        await require_auth(authorization)
        return await realtime_service.usage_report()

    @app.websocket("/v1/realtime")
    async def realtime_websocket(websocket: WebSocket) -> None:
        await realtime_service.handle_websocket(websocket)

    @app.get("/metrics")
    async def metrics() -> Response:
        await service_lifecycle.observe_gpu_memory()
        return Response(
            content=service_lifecycle.telemetry.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("services/gpu-service/config/default.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

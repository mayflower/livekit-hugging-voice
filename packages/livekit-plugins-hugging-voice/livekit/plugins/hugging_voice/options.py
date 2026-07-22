"""Strict public connection and authentication option resolution."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse


def resolve_base_urls(
    *,
    base_url: str | None,
    base_urls: Sequence[str] | None,
) -> tuple[str, ...]:
    if base_url is not None and base_urls is not None:
        raise ValueError("provide base_url or base_urls, not both")
    values: Sequence[str]
    if base_url is not None:
        values = [base_url]
    elif base_urls is not None:
        values = base_urls
    else:
        configured = os.environ.get("HUGGING_VOICE_BASE_URL")
        if not configured:
            raise ValueError("base_url is required (or set HUGGING_VOICE_BASE_URL)")
        values = [configured]
    if not values or len(values) > 16:
        raise ValueError("base_urls must contain between one and sixteen endpoints")
    normalized: list[str] = []
    for value in values:
        endpoint = value.strip().rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError(f"invalid Hugging Voice WebSocket URL: {value!r}")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("Hugging Voice URLs cannot contain credentials, query, or fragment")
        if parsed.path in {"", "/"}:
            endpoint += "/v1/realtime"
        elif parsed.path != "/v1/realtime":
            raise ValueError("Hugging Voice URL path must be /v1/realtime")
        if endpoint not in normalized:
            normalized.append(endpoint)
    return tuple(normalized)


def resolve_token(*, token: str | None, token_file: str | Path | None) -> str:
    if token is not None and token_file is not None:
        raise ValueError("provide token or token_file, not both")
    candidate = token
    path: Path | None = Path(token_file) if token_file is not None else None
    if candidate is None and path is None:
        candidate = os.environ.get("HUGGING_VOICE_TOKEN")
        configured_path = os.environ.get("HUGGING_VOICE_TOKEN_FILE")
        if candidate is not None and configured_path is not None:
            raise ValueError("set only one of HUGGING_VOICE_TOKEN and HUGGING_VOICE_TOKEN_FILE")
        if configured_path is not None:
            path = Path(configured_path)
    if path is not None:
        try:
            candidate = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"unable to read Hugging Voice token file {path}") from exc
    if candidate is None:
        raise ValueError("token is required (or set HUGGING_VOICE_TOKEN/HUGGING_VOICE_TOKEN_FILE)")
    if not candidate or len(candidate) > 4_096 or any(char.isspace() for char in candidate):
        raise ValueError(
            "Hugging Voice token must be non-empty, bounded, and contain no whitespace"
        )
    return candidate

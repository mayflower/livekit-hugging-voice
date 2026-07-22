"""Bearer authentication from a mounted secret file."""

from __future__ import annotations

import hmac
from pathlib import Path


class AuthenticationError(ValueError):
    pass


class TokenAuthenticator:
    def __init__(self, token: str) -> None:
        if not token or len(token) > 4_096 or any(character.isspace() for character in token):
            raise AuthenticationError(
                "bearer token must be non-empty, bounded, and contain no whitespace"
            )
        self._token = token.encode("utf-8")

    @classmethod
    def from_file(cls, path: Path) -> TokenAuthenticator:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise AuthenticationError(f"unable to read bearer token file {path}") from exc
        return cls(token)

    def authenticate_header(self, authorization: str | None) -> bool:
        if (
            authorization is None
            or len(authorization) > 4_103
            or not authorization.startswith("Bearer ")
        ):
            return False
        candidate = authorization.removeprefix("Bearer ").encode("utf-8")
        return hmac.compare_digest(candidate, self._token)

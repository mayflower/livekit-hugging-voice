"""Per-session generation-tagged cancellation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationToken:
    session_id: str
    turn_id: str
    turn_revision: int
    generation_id: str
    epoch: int


class GenerationCancellation:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._epoch = 0
        self._current: GenerationToken | None = None

    @property
    def current(self) -> GenerationToken | None:
        return self._current

    def start(
        self,
        *,
        turn_id: str,
        turn_revision: int,
        generation_id: str,
    ) -> GenerationToken:
        self._epoch += 1
        token = GenerationToken(
            session_id=self._session_id,
            turn_id=turn_id,
            turn_revision=turn_revision,
            generation_id=generation_id,
            epoch=self._epoch,
        )
        self._current = token
        return token

    def cancel(self, generation_id: str | None = None) -> GenerationToken | None:
        current = self._current
        if current is None:
            return None
        if generation_id is not None and generation_id != current.generation_id:
            return None
        self._epoch += 1
        self._current = None
        return current

    def finish(self, token: GenerationToken) -> bool:
        if not self.is_current(token):
            return False
        self._current = None
        return True

    def is_current(self, token: GenerationToken) -> bool:
        return self._current == token and token.epoch == self._epoch

    def reset(self) -> None:
        self._epoch += 1
        self._current = None

"""Metrics that exist so an operator can tell two deployments apart."""

from __future__ import annotations

from hugging_voice_service.config import ServiceSettings
from hugging_voice_service.telemetry import ServiceTelemetry
from hugging_voice_service.version import __version__


def test_build_info_publishes_the_latency_relevant_settings() -> None:
    """A dashboard must be able to tell which configuration produced a run.

    The llama.cpp cache flags are the difference between a reused prompt prefix
    and a full re-prefill on every turn, so a metrics scrape that cannot name
    them leaves two deployments indistinguishable in Grafana.
    """

    telemetry = ServiceTelemetry()
    telemetry.describe_build(ServiceSettings())
    rendered = telemetry.render().decode("utf-8")

    line = next(row for row in rendered.splitlines() if row.startswith("hugging_voice_build_info{"))
    assert line.endswith(" 1.0")
    for label in (
        f'version="{__version__}"',
        'llm_profile="compat_gemma31"',
        'swa_full="false"',
        'context_size="32768"',
        'cache_type_k="f16"',
        'cache_reuse="0"',
        'partial_stt="false"',
    ):
        assert label in line, label

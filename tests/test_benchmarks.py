from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

REPO_ROOT = Path(__file__).parents[1]


def _load_summary_module() -> ModuleType:
    path = REPO_ROOT / "benchmarks" / "summarize.py"
    spec = importlib.util.spec_from_file_location("benchmark_summarize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_summary_uses_only_measured_values(tmp_path: Path) -> None:
    module = _load_summary_module()
    summarize_records = cast(object, module.summarize_records)
    report = summarize_records(  # type: ignore[operator]
        [
            {"record_type": "metadata", "metadata": {"gpu": "measured-device"}},
            {
                "record_type": "turn",
                "metrics": {"speech_stop_to_first_audio_frame_seconds": 1.0},
            },
            {
                "record_type": "turn",
                "metrics": {"speech_stop_to_first_audio_frame_seconds": 3.0},
            },
            {"record_type": "error", "message": "CUDA OOM"},
        ],
        tmp_path / "raw.jsonl",
    )
    metric = report["metrics"]["speech_stop_to_first_audio_frame_seconds"]
    assert metric == {"count": 2, "p50": 2.0, "p95": 2.9, "p99": 2.98, "min": 1.0, "max": 3.0}
    assert report["turns"] == 2
    assert report["errors"] == 1
    assert report["ooms"] == 1


def test_empty_benchmark_does_not_invent_targets(tmp_path: Path) -> None:
    module = _load_summary_module()
    report = module.summarize_records([], tmp_path / "empty.jsonl")
    markdown = module.render_markdown(report)
    assert report["metrics"] == {}
    assert "No latency observations were recorded" in markdown


def test_prometheus_summary_uses_snapshot_deltas(tmp_path: Path) -> None:
    module = _load_summary_module()
    before = """\
hugging_voice_stt_queue_seconds_bucket{le="0.1"} 1
hugging_voice_stt_queue_seconds_bucket{le="+Inf"} 1
hugging_voice_stt_queue_seconds_count 1
hugging_voice_stt_queue_seconds_sum 0.05
"""
    after = """\
hugging_voice_stt_queue_seconds_bucket{le="0.1"} 2
hugging_voice_stt_queue_seconds_bucket{le="0.5"} 3
hugging_voice_stt_queue_seconds_bucket{le="+Inf"} 3
hugging_voice_stt_queue_seconds_count 3
hugging_voice_stt_queue_seconds_sum 0.65
hugging_voice_model_loads_total{model="gemma"} 1
"""
    report = module.summarize_records(
        [
            {"record_type": "prometheus", "phase": "before", "text": before},
            {"record_type": "prometheus", "phase": "after", "text": after},
        ],
        tmp_path / "raw.jsonl",
    )
    service = report["service_metrics"]
    assert service["histograms"]["hugging_voice_stt_queue_seconds"]["count"] == 2
    assert service["histograms"]["hugging_voice_stt_queue_seconds"]["mean"] == 0.3
    assert service["model_loads"] == {"gemma": 1}

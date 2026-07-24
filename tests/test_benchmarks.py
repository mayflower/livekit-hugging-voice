from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

REPO_ROOT = Path(__file__).parents[1]


def _load_summary_module() -> ModuleType:
    path = REPO_ROOT / "benchmarks" / "summarize.py"
    spec = importlib.util.spec_from_file_location("benchmark_summarize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_module() -> ModuleType:
    path = REPO_ROOT / "benchmarks" / "multisession_soak.py"
    spec = importlib.util.spec_from_file_location("benchmark_multisession", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_tool_error_rate_and_concurrency_groups_are_specific(tmp_path: Path) -> None:
    module = _load_summary_module()
    report = module.summarize_records(
        [
            {
                "record_type": "metadata",
                "metadata": {"session_concurrency": 2},
            },
            {
                "record_type": "turn",
                "tool_call_emitted_at": 1.0,
                "metrics": {"speech_stop_to_tool_call_seconds": 0.5},
            },
            {"record_type": "error", "message": "unrelated transport failure"},
            {
                "record_type": "error",
                "message": "tool turn failed",
                "tool_call_error": True,
            },
        ],
        tmp_path / "raw.jsonl",
    )
    assert report["tool_call_error_rate"] == 0.5
    assert (
        report["metrics_by_session_concurrency"]["2"]["speech_stop_to_tool_call_seconds"]["p50"]
        == 0.5
    )


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
llamacpp:prompt_tokens_total 100
llamacpp:prompt_seconds_total 2.5
llamacpp:n_decode_total 25
llamacpp:n_busy_slots_per_decode 1.6
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
    assert service["llama_cpp"] == {
        "llamacpp:prompt_tokens_total": 100.0,
        "llamacpp:prompt_seconds_total": 2.5,
        "llamacpp:n_decode_total": 25.0,
        "llamacpp:n_busy_slots_per_decode": 1.6,
    }


def test_multisession_seed_and_mixed_workload_are_deterministic() -> None:
    module = _load_runner_module()
    assert module.build_canaries(8, 42) == module.build_canaries(8, 42)
    assert len(set(module.build_canaries(8, 42))) == 8
    assert [module.select_turn_type("mixed", index, 1) for index in range(4)] == [
        "normal",
        "tool",
        "normal",
        "tool",
    ]
    assert [module.select_turn_type("mixed", 0, turn) for turn in range(1, 5)] == [
        "normal",
        "tool",
        "normal",
        "tool",
    ]


def test_summary_reports_per_session_fairness_and_turn_types(tmp_path: Path) -> None:
    module = _load_summary_module()
    metadata = {
        "profile_id": "profile-a",
        "configuration_fingerprint": "d" * 64,
        "session_concurrency": 2,
        "arrival_mode": "barrier",
        "workload": "mixed",
        "seed": 7,
        "gpu": "gpu",
        "cuda_runtime": "12.8",
        "git_commit": "a" * 40,
        "container_image_digest": "sha256:" + "b" * 64,
        "service_models": {"llm": "test"},
        "wav_sha256": ["c" * 64],
    }
    records: list[dict[str, object]] = [{"record_type": "metadata", "metadata": metadata}]
    for session_index, value in ((0, 1.0), (1, 2.0)):
        records.append(
            {
                "record_type": "turn",
                "profile_id": "profile-a",
                "session_concurrency": 2,
                "session_index": session_index,
                "session_id": f"session_{session_index}",
                "llama_slot_id": session_index,
                "turn_type": "tool" if session_index else "normal",
                "status": "completed",
                "audio_chunk_count": 2,
                "metrics": {"speech_stop_to_first_audio_frame_seconds": value},
            }
        )
    records.append({"record_type": "run_complete", "actual_duration_seconds": 60.0})
    report = module.summarize_records(records, tmp_path / "raw.jsonl")
    assert report["provenance_complete"] is True
    assert report["correctness_passed"] is True
    assert report["turns_per_minute"] == 2.0
    assert report["fairness"]["slowest_to_median_p95_ratio"] == 4 / 3
    assert set(report["metrics_by_session"]) == {"0", "1"}
    assert set(report["metrics_by_turn_type"]) == {"normal", "tool"}
    assert set(report["metrics_by_profile"]) == {"profile-a"}


def test_summary_fails_correctness_for_leak_and_truncated_audio(tmp_path: Path) -> None:
    module = _load_summary_module()
    report = module.summarize_records(
        [
            {
                "record_type": "turn",
                "session_index": 0,
                "turn_index": 1,
                "status": "completed",
                "audio_chunk_count": 1,
                "cross_session_leak": True,
                "metrics": {},
            }
        ],
        tmp_path / "raw.jsonl",
    )
    assert report["correctness_passed"] is False
    assert any("cross_session_leak" in item for item in report["correctness_violations"])
    assert any("incomplete_audio" in item for item in report["correctness_violations"])


def test_summary_marks_incomplete_provenance(tmp_path: Path) -> None:
    module = _load_summary_module()
    report = module.summarize_records(
        [{"record_type": "metadata", "metadata": {"profile_id": "only-field"}}],
        tmp_path / "raw.jsonl",
    )
    assert report["provenance_complete"] is False
    assert "gpu" in report["missing_provenance"]

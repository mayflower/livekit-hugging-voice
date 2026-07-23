#!/usr/bin/env python3
"""Turn raw, measured benchmark JSONL into JSON and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CLIENT_METRICS = {
    "speech_stop_to_final_transcript_seconds",
    "speech_stop_to_first_text_delta_seconds",
    "speech_stop_to_first_audio_frame_seconds",
    "barge_in_to_last_old_audio_frame_seconds",
    "response_duration_seconds",
    "speech_stop_to_tool_call_seconds",
    "tool_duration_seconds",
    "tool_result_ack_to_final_first_text_seconds",
    "tool_result_ack_to_final_first_audio_seconds",
    "speech_stop_to_final_first_audio_seconds",
}
SERVICE_HISTOGRAMS = {
    "hugging_voice_stt_queue_seconds",
    "hugging_voice_stt_inference_seconds",
    "hugging_voice_transcription_delay_seconds",
    "hugging_voice_llm_ttft_seconds",
    "hugging_voice_llm_duration_seconds",
    "hugging_voice_llm_tokens_per_second",
    "hugging_voice_tts_queue_seconds",
    "hugging_voice_tts_ttfa_seconds",
    "hugging_voice_tts_duration_seconds",
    "hugging_voice_tts_audio_seconds",
    "hugging_voice_first_audio_latency_seconds",
    "hugging_voice_barge_in_stop_latency_seconds",
    "hugging_voice_tool_decision_seconds",
    "hugging_voice_tool_result_wait_seconds",
    "hugging_voice_tool_result_to_first_text_seconds",
    "hugging_voice_tool_result_to_first_audio_seconds",
}
BUCKET_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_bucket\{le="(?P<le>[^"]+)"\}$')
LOAD_RE = re.compile(r'^hugging_voice_model_loads_total\{model="(?P<model>[^"]+)"\}$')


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for measured observations."""
    if not values:
        raise ValueError("cannot calculate a percentile without observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def prometheus_samples(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            samples[key] = value
    return samples


def histogram_quantile(buckets: list[tuple[float, float]], count: float, q: float) -> float | None:
    target = count * q
    previous_upper = 0.0
    previous_count = 0.0
    for upper, cumulative in sorted(buckets):
        if cumulative < target:
            previous_upper = upper
            previous_count = cumulative
            continue
        if math.isinf(upper):
            return previous_upper if previous_count > 0 else None
        in_bucket = cumulative - previous_count
        if in_bucket <= 0:
            return upper
        fraction = (target - previous_count) / in_bucket
        return previous_upper + (upper - previous_upper) * fraction
    return None


def summarize_prometheus(records: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = {
        str(record.get("phase")): prometheus_samples(str(record.get("text", "")))
        for record in records
        if record.get("record_type") == "prometheus"
    }
    before = snapshots.get("before", {})
    after = snapshots.get("after", {})
    if not after:
        return {}
    result: dict[str, Any] = {"histograms": {}, "model_loads": {}}
    for name in sorted(SERVICE_HISTOGRAMS):
        count = after.get(f"{name}_count", 0.0) - before.get(f"{name}_count", 0.0)
        total = after.get(f"{name}_sum", 0.0) - before.get(f"{name}_sum", 0.0)
        if count <= 0:
            continue
        buckets: list[tuple[float, float]] = []
        for key, after_value in after.items():
            match = BUCKET_RE.match(key)
            if match is None or match.group("name") != name:
                continue
            raw_upper = match.group("le")
            upper = math.inf if raw_upper == "+Inf" else float(raw_upper)
            delta = after_value - before.get(key, 0.0)
            buckets.append((upper, delta))
        result["histograms"][name] = {
            "count": int(count),
            "mean": total / count,
            "p50_bucket_estimate": histogram_quantile(buckets, count, 0.50),
            "p95_bucket_estimate": histogram_quantile(buckets, count, 0.95),
            "p99_bucket_estimate": histogram_quantile(buckets, count, 0.99),
        }
    for key, value in after.items():
        match = LOAD_RE.match(key)
        if match is not None:
            result["model_loads"][match.group("model")] = int(value)
    tts_duration = after.get("hugging_voice_tts_duration_seconds_sum", 0.0) - before.get(
        "hugging_voice_tts_duration_seconds_sum", 0.0
    )
    tts_audio = after.get("hugging_voice_tts_audio_seconds_sum", 0.0) - before.get(
        "hugging_voice_tts_audio_seconds_sum", 0.0
    )
    result["tts_aggregate_rtf"] = tts_duration / tts_audio if tts_audio > 0 else None
    return result


def summarize_gpu_csv(paths: list[Path]) -> dict[str, dict[str, float | int]]:
    phases: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                phase = row.get("phase", "").strip()
                raw = row.get("memory.used", "").strip()
                if phase and raw:
                    phases[phase].append(float(raw))
    return {
        phase: {
            "count": len(values),
            "memory_used_mib_p50": percentile(values, 0.50),
            "memory_used_mib_p95": percentile(values, 0.95),
            "memory_used_mib_max": max(values),
        }
        for phase, values in sorted(phases.items())
        if values
    }


def summarize_records(records: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = next(
        (
            record.get("metadata", {})
            for record in records
            if record.get("record_type") == "metadata"
        ),
        {},
    )
    observations: dict[str, list[float]] = defaultdict(list)
    observations_by_concurrency: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    errors: list[dict[str, Any]] = []
    turns = 0
    tool_turns = 0
    stale = 0
    cancelled = 0
    tool_errors = 0
    default_concurrency = metadata.get("session_concurrency")
    for record in records:
        kind = record.get("record_type")
        if kind == "turn":
            turns += 1
            if record.get("tool_call_emitted_at") is not None:
                tool_turns += 1
            stale += int(record.get("stale_count", 0))
            cancelled += int(bool(record.get("cancelled", False)))
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                if (
                    name in CLIENT_METRICS
                    and isinstance(value, int | float)
                    and math.isfinite(value)
                ):
                    observations[name].append(float(value))
                    concurrency = record.get("session_concurrency", default_concurrency)
                    if isinstance(concurrency, int) and concurrency in {1, 2}:
                        observations_by_concurrency[concurrency][name].append(float(value))
        elif kind == "error":
            errors.append(record)
            tool_errors += int(bool(record.get("tool_call_error", False)))

    def metric_summary(
        source_observations: dict[str, list[float]],
    ) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
                "min": min(values),
                "max": max(values),
            }
            for name, values in sorted(source_observations.items())
        }

    successful_or_failed_tool_turns = tool_turns + tool_errors

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "metadata": metadata,
        "turns": turns,
        "tool_turns": tool_turns,
        "tool_call_error_rate": (
            tool_errors / successful_or_failed_tool_turns
            if successful_or_failed_tool_turns
            else None
        ),
        "stale": stale,
        "cancelled": cancelled,
        "errors": len(errors),
        "ooms": sum("oom" in str(error).lower() for error in errors),
        "metrics": metric_summary(observations),
        "metrics_by_session_concurrency": {
            str(concurrency): metric_summary(grouped)
            for concurrency, grouped in sorted(observations_by_concurrency.items())
        },
        "service_metrics": summarize_prometheus(records),
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# Hugging Voice benchmark report",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Raw source: `{report['source']}`  ",
        f"Measured turns: **{report['turns']}**  ",
        f"Errors: **{report['errors']}**; OOMs: **{report['ooms']}**",
        "",
        "## Provenance",
        "",
    ]
    if metadata:
        for key in sorted(metadata):
            value = metadata[key]
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lines.append(f"- `{key}`: `{rendered}`")
    else:
        lines.append(
            "No provenance metadata was present; this run is incomplete and not release evidence."
        )

    lines.extend(["", "## Measurements", ""])
    metrics = report["metrics"]
    if not metrics:
        lines.append(
            "No latency observations were recorded. No target values are shown as results."
        )
    else:
        lines.extend(
            [
                "| Metric | n | p50 | p95 | p99 | min | max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in metrics.items():
            lines.append(
                f"| `{name}` | {values['count']} | {values['p50']:.4f} | "
                f"{values['p95']:.4f} | {values['p99']:.4f} | "
                f"{values['min']:.4f} | {values['max']:.4f} |"
            )
        lines.append("")
        lines.append("All time values are seconds and come from the raw observations.")
    grouped_metrics = report.get("metrics_by_session_concurrency", {})
    if grouped_metrics:
        lines.extend(["", "## Measurements by session concurrency", ""])
        for concurrency, concurrency_metrics in grouped_metrics.items():
            lines.extend(
                [
                    f"### {concurrency} active session(s)",
                    "",
                    "| Metric | n | p50 | p95 | p99 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for name, values in concurrency_metrics.items():
                lines.append(
                    f"| `{name}` | {values['count']} | {values['p50']:.4f} | "
                    f"{values['p95']:.4f} | {values['p99']:.4f} |"
                )
            lines.append("")
    service_metrics = report.get("service_metrics", {})
    if service_metrics:
        lines.extend(["", "## Service metrics", ""])
        lines.append(
            "Prometheus percentile values are interpolated estimates from measured histogram "
            "buckets; means and aggregate TTS RTF use counter deltas."
        )
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(service_metrics, indent=2, sort_keys=True))
        lines.append("```")
    gpu_memory = report.get("gpu_memory", {})
    if gpu_memory:
        lines.extend(["", "## GPU memory", "", "```json"])
        lines.append(json.dumps(gpu_memory, indent=2, sort_keys=True))
        lines.append("```")
    return "\n".join(lines) + "\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, help="JSONL emitted by the soak/E2E runner")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/reports"))
    parser.add_argument("--gpu-csv", action="append", type=Path, default=[])
    args = parser.parse_args()
    records = load_jsonl(args.raw)
    report = summarize_records(records, args.raw)
    report["gpu_memory"] = summarize_gpu_csv(args.gpu_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw.stem.removesuffix(".raw")
    json_path = args.output_dir / f"{stem}.summary.json"
    markdown_path = args.output_dir / f"{stem}.summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resolve version-pinned base images to immutable release digests."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

IMAGES = (
    "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
    "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
    "ubuntu:24.04",
    "python:3.11.13-slim-bookworm",
    "ghcr.io/astral-sh/uv:0.11.7",
    "livekit/livekit-server:v1.13.4",
)


def resolve(image: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image,
            "--format",
            "{{json .Manifest.Digest}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = json.loads(result.stdout)
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError(f"Docker returned an invalid digest for {image}: {digest!r}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite release evidence: {args.output}")
    report = {
        "schema_version": 1,
        "resolved_at": datetime.now(UTC).isoformat(),
        "images": {image: resolve(image) for image in IMAGES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

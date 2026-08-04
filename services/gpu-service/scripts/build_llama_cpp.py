"""Build the pinned llama-server from an already checked-out source tree."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

LLAMA_CPP_COMMIT = "3ce7da2c852c538c4c5f9806da27029cf8c9cc4a"
# Must stay in sync with LLAMA_CUDA_ARCHITECTURES in services/gpu-service/Dockerfile
# so a locally built llama-server runs on the same GPUs as the published image.
CUDA_ARCHITECTURES = "86;89;120"


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build(source: Path, build_dir: Path, *, jobs: int, cuda_architectures: str) -> Path:
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != LLAMA_CPP_COMMIT:
        raise RuntimeError(
            f"llama.cpp checkout mismatch: expected={LLAMA_CPP_COMMIT} actual={actual_commit}"
        )
    run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=ON",
            "-DGGML_NATIVE=OFF",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=source,
    )
    run(
        ["cmake", "--build", str(build_dir), "--target", "llama-server", "--parallel", str(jobs)],
        cwd=source,
    )
    binary = build_dir / "bin" / "llama-server"
    if not binary.is_file():
        raise RuntimeError(f"build completed without {binary}")
    return binary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cuda-architectures", default=CUDA_ARCHITECTURES)
    args = parser.parse_args()
    print(
        build(
            args.source.resolve(),
            args.build.resolve(),
            jobs=args.jobs,
            cuda_architectures=args.cuda_architectures,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

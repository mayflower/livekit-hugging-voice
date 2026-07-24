"""Explicit Hugging Face model prefetch with atomic SHA-256 lock creation."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import HfApi, hf_hub_download

from .model_manifest import (
    HuggingFaceModelSpec,
    LockedFile,
    LockedModel,
    ManifestError,
    ModelLock,
    ModelManifest,
    PythonPackageModelSpec,
    load_manifest,
    render_lock,
    sha256_file,
)


class ModelInfo(Protocol):
    sha: str | None


class ModelInfoClient(Protocol):
    def model_info(self, repo_id: str, *, revision: str) -> ModelInfo: ...


DownloadFile = Callable[..., str]


def prefetch_models(
    manifest: ModelManifest,
    *,
    output_root: Path,
    lock_path: Path,
    api: ModelInfoClient | None = None,
    download_file: DownloadFile | None = None,
) -> ModelLock:
    """Download every allowlisted file, then atomically publish one complete lock."""
    info_client = api or cast(ModelInfoClient, HfApi())
    downloader = download_file or cast(DownloadFile, hf_hub_download)
    output_root.mkdir(parents=True, exist_ok=True)
    locked_models: list[LockedModel] = []

    for model in manifest.models:
        if isinstance(model, PythonPackageModelSpec):
            locked_models.append(_lock_python_package(model))
            continue
        locked_models.append(
            _prefetch_huggingface_model(
                model,
                output_root=output_root,
                api=info_client,
                download_file=downloader,
            )
        )

    lock = ModelLock(profile_id=manifest.profile_id, models=tuple(locked_models))
    _atomic_write(lock_path, render_lock(lock))
    return lock


def _prefetch_huggingface_model(
    model: HuggingFaceModelSpec,
    *,
    output_root: Path,
    api: ModelInfoClient,
    download_file: DownloadFile,
) -> LockedModel:
    info = api.model_info(model.source_repo, revision=model.revision)
    if info.sha is None or len(info.sha) != 40:
        raise ManifestError(f"unable to resolve exact commit for {model.source_repo}")
    if info.sha != model.revision:
        raise ManifestError(
            f"revision mismatch for {model.source_repo}: "
            f"expected={model.revision} resolved={info.sha}"
        )

    local_dir = output_root / Path(model.id)
    local_dir.mkdir(parents=True, exist_ok=True)
    files: list[LockedFile] = []
    for file_spec in model.files:
        downloaded = Path(
            download_file(
                repo_id=model.source_repo,
                filename=file_spec.path,
                revision=info.sha,
                local_dir=str(local_dir),
            )
        )
        expected = (local_dir / file_spec.path).resolve()
        if downloaded.resolve() != expected or not expected.is_file():
            raise ManifestError(
                f"download for {model.source_repo}:{file_spec.path} did not produce {expected}"
            )
        files.append(
            LockedFile(
                path=file_spec.path,
                size=expected.stat().st_size,
                sha256=sha256_file(expected),
            )
        )
    return LockedModel(
        delivery="huggingface",
        id=model.id,
        source_repo=model.source_repo,
        revision=info.sha,
        files=tuple(files),
        license=model.license,
    )


def _lock_python_package(model: PythonPackageModelSpec) -> LockedModel:
    return LockedModel(
        delivery="python-package",
        id=model.id,
        source_repo=model.source_repo,
        revision=model.revision,
        files=(),
        license=model.license,
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch pinned Hugging Voice model files")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock", type=Path, default=Path("models/manifest.lock.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    lock = prefetch_models(
        load_manifest(args.manifest),
        output_root=args.output,
        lock_path=args.lock,
    )
    print(f"prefetched {len(lock.models)} model entries and wrote {args.lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Model manifest parsing and fully offline lock verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
PACKAGE_VERSION_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.-]+)?$"
MODEL_ID_PATTERN = r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"


class ManifestError(ValueError):
    """Raised for malformed, unsafe, incomplete, or mismatched model manifests."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelFileSpec(StrictModel):
    path: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_relative_path(self) -> ModelFileSpec:
        _safe_relative_path(self.path)
        return self


class HuggingFaceModelSpec(StrictModel):
    delivery: Literal["huggingface"]
    id: str = Field(pattern=MODEL_ID_PATTERN, max_length=256)
    source_repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=COMMIT_PATTERN)
    license: str = Field(min_length=1, max_length=128)
    files: tuple[ModelFileSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_files(self) -> HuggingFaceModelSpec:
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("model file paths must be unique")
        return self


class PythonPackageModelSpec(StrictModel):
    delivery: Literal["python-package"]
    id: str = Field(pattern=MODEL_ID_PATTERN, max_length=256)
    source_repo: str = Field(pattern=r"^pypi:[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=PACKAGE_VERSION_PATTERN)
    license: str = Field(min_length=1, max_length=128)
    files: tuple[ModelFileSpec, ...] = ()


ModelSpec = Annotated[
    HuggingFaceModelSpec | PythonPackageModelSpec, Field(discriminator="delivery")
]


class ModelManifest(StrictModel):
    schema_version: Literal[1] = 1
    models: tuple[ModelSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_models(self) -> ModelManifest:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        return self


class LockedFile(StrictModel):
    path: str = Field(min_length=1, max_length=512)
    size: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_relative_path(self) -> LockedFile:
        _safe_relative_path(self.path)
        return self


class LockedModel(StrictModel):
    delivery: Literal["huggingface", "python-package"]
    id: str = Field(pattern=MODEL_ID_PATTERN, max_length=256)
    source_repo: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    files: tuple[LockedFile, ...]
    license: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_revision_and_files(self) -> LockedModel:
        pattern = COMMIT_PATTERN if self.delivery == "huggingface" else PACKAGE_VERSION_PATTERN
        if re.fullmatch(pattern, self.revision) is None:
            raise ValueError(f"invalid {self.delivery} revision: {self.revision}")
        if self.delivery == "huggingface" and not self.files:
            raise ValueError("Hugging Face lock entries require at least one file")
        if self.delivery == "python-package" and self.files:
            raise ValueError("Python package lock entries must not contain artifact files")
        return self


class ModelLock(StrictModel):
    schema_version: Literal[1] = 1
    models: tuple[LockedModel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_models(self) -> ModelLock:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("locked model ids must be unique")
        return self


MODEL_MANIFEST_ADAPTER = TypeAdapter(ModelManifest)
MODEL_LOCK_ADAPTER = TypeAdapter(ModelLock)


def load_manifest(path: Path | str) -> ModelManifest:
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        return MODEL_MANIFEST_ADAPTER.validate_python(raw)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"invalid model manifest {manifest_path}: {exc}") from exc


def load_lock(path: Path | str) -> ModelLock:
    lock_path = Path(path)
    try:
        return MODEL_LOCK_ADAPTER.validate_json(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"invalid model lock {lock_path}: {exc}") from exc


def verify_lock(lock: ModelLock, root: Path | str) -> None:
    """Verify every locked artifact from local bytes without network access."""
    model_root = Path(root).resolve()
    failures: list[str] = []
    for model in lock.models:
        if model.delivery == "python-package":
            package_name = model.source_repo.removeprefix("pypi:")
            try:
                installed_version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                failures.append(f"missing Python package: {package_name}=={model.revision}")
            else:
                if installed_version != model.revision:
                    failures.append(
                        f"package version mismatch: {package_name} "
                        f"expected={model.revision} actual={installed_version}"
                    )
            continue
        for file in model.files:
            artifact = _artifact_path(model_root, model.id, file.path)
            if not artifact.is_file():
                failures.append(f"missing: {artifact}")
                continue
            stat = artifact.stat()
            if stat.st_size != file.size:
                failures.append(
                    f"size mismatch: {artifact} expected={file.size} actual={stat.st_size}"
                )
                continue
            actual_sha = sha256_file(artifact)
            if actual_sha != file.sha256:
                failures.append(
                    f"sha256 mismatch: {artifact} expected={file.sha256} actual={actual_sha}"
                )
    if failures:
        raise ManifestError("model verification failed:\n" + "\n".join(failures))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_lock(lock: ModelLock) -> str:
    return json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe model file path: {value!r}")
    return path


def _artifact_path(root: Path, model_id: str, relative: str) -> Path:
    model_path = _safe_relative_path(model_id)
    file_path = _safe_relative_path(relative)
    candidate = (root / model_path / file_path).resolve()
    if not candidate.is_relative_to(root):
        raise ManifestError(f"model artifact escapes root: {candidate}")
    return candidate


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a local Hugging Voice model lock")
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    verify_lock(load_lock(args.lock), args.root)
    print(f"verified model lock {args.lock} under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

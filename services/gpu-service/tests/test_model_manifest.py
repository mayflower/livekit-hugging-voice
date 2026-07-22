from __future__ import annotations

import json
from pathlib import Path

import pytest
from hugging_voice_service.model_manifest import (
    ManifestError,
    ModelLock,
    load_lock,
    load_manifest,
    verify_lock,
)
from hugging_voice_service.model_manifest import (
    main as verify_main,
)
from hugging_voice_service.model_prefetch import ModelInfo, prefetch_models
from hugging_voice_service.model_prefetch import main as prefetch_main

REPO_ROOT = Path(__file__).parents[3]
REVISION = "a" * 40


class FakeInfo:
    sha: str | None = REVISION


class FakeApi:
    def model_info(self, repo_id: str, *, revision: str) -> ModelInfo:
        assert repo_id == "owner/model"
        assert revision == REVISION
        return FakeInfo()


def write_manifest(path: Path, *, file_path: str = "weights.bin") -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "models:",
                "  - delivery: huggingface",
                "    id: logical/model",
                "    source_repo: owner/model",
                f"    revision: {REVISION}",
                "    license: Apache-2.0",
                "    files:",
                f"      - path: {file_path}",
            ]
        ),
        encoding="utf-8",
    )


def fake_download(**kwargs: object) -> str:
    local_dir = Path(str(kwargs["local_dir"]))
    filename = str(kwargs["filename"])
    destination = local_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"real fixture bytes")
    return str(destination)


def test_real_manifest_has_exact_revisions_and_expected_models() -> None:
    manifest = load_manifest(REPO_ROOT / "models" / "manifest.yaml")
    assert {model.id for model in manifest.models} == {
        "google/gemma-4-31B-it",
        "nvidia/parakeet-tdt-0.6b-v3",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "silero-vad",
    }
    hf_models = [model for model in manifest.models if model.delivery == "huggingface"]
    assert all(len(model.revision) == 40 for model in hf_models)
    gemma = next(model for model in hf_models if model.id == "google/gemma-4-31B-it")
    assert [file.path for file in gemma.files] == ["gemma-4-31B-it-Q4_0.gguf"]


def test_prefetch_writes_atomic_lock_and_offline_verify_detects_changes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    output = tmp_path / "models"
    lock_path = tmp_path / "manifest.lock.json"
    write_manifest(manifest_path)
    lock = prefetch_models(
        load_manifest(manifest_path),
        output_root=output,
        lock_path=lock_path,
        api=FakeApi(),
        download_file=fake_download,
    )
    assert load_lock(lock_path) == lock
    assert lock_path.stat().st_mode & 0o777 == 0o644
    verify_lock(lock, output)

    (output / "logical" / "model" / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ManifestError, match="size mismatch"):
        verify_lock(lock, output)


def test_failure_does_not_publish_partial_lock(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    lock_path = tmp_path / "manifest.lock.json"
    write_manifest(manifest_path)
    lock_path.write_text("original", encoding="utf-8")

    def fail_download(**kwargs: object) -> str:
        raise OSError(f"deliberate download failure for {kwargs['filename']}")

    with pytest.raises(OSError, match="deliberate"):
        prefetch_models(
            load_manifest(manifest_path),
            output_root=tmp_path / "models",
            lock_path=lock_path,
            api=FakeApi(),
            download_file=fail_download,
        )
    assert lock_path.read_text(encoding="utf-8") == "original"


def test_manifest_rejects_branch_revision_and_path_traversal(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest_path)
    text = manifest_path.read_text(encoding="utf-8").replace(REVISION, "main")
    manifest_path.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(manifest_path)

    write_manifest(manifest_path, file_path="../outside")
    with pytest.raises(ManifestError, match="unsafe model file path"):
        load_manifest(manifest_path)


def test_lock_rejects_placeholder_hash_and_size(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    payload = {
        "schema_version": 1,
        "models": [
            {
                "delivery": "huggingface",
                "id": "logical/model",
                "source_repo": "owner/model",
                "revision": REVISION,
                "files": [{"path": "weights.bin", "size": 0, "sha256": "placeholder"}],
                "license": "Apache-2.0",
            }
        ],
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_lock(lock_path)


def test_package_lock_checks_installed_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = ModelLock.model_validate(
        {
            "schema_version": 1,
            "models": [
                {
                    "delivery": "python-package",
                    "id": "silero-vad",
                    "source_repo": "pypi:silero-vad",
                    "revision": "6.2.1",
                    "files": [],
                    "license": "MIT",
                }
            ],
        }
    )
    monkeypatch.setattr("importlib.metadata.version", lambda package: "6.2.1")
    verify_lock(lock, tmp_path)
    monkeypatch.setattr("importlib.metadata.version", lambda package: "6.2.0")
    with pytest.raises(ManifestError, match="package version mismatch"):
        verify_lock(lock, tmp_path)


def test_prefetch_and_verify_cli_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    output = tmp_path / "models"
    lock_path = tmp_path / "manifest.lock.json"
    write_manifest(manifest_path)
    monkeypatch.setattr("hugging_voice_service.model_prefetch.HfApi", lambda: FakeApi())
    monkeypatch.setattr("hugging_voice_service.model_prefetch.hf_hub_download", fake_download)

    assert (
        prefetch_main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--lock",
                str(lock_path),
            ]
        )
        == 0
    )
    assert verify_main(["--lock", str(lock_path), "--root", str(output)]) == 0

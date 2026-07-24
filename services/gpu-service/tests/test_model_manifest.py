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
                "profile_id: test_profile",
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
    assert manifest.profile_id == "compat_gemma31_qwen17_ggml"
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
    qwen = next(model for model in hf_models if model.id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    # The shipped manifest must cover the default voice_clone mode.
    assert [file.path for file in qwen.files] == [
        "qwen-talker-1.7b-base-BF16.gguf",
        "qwen-tokenizer-12hz-BF16.gguf",
    ]


def test_cuda_tts_profile_has_an_exact_explicit_offline_lock() -> None:
    profile_dir = REPO_ROOT / "models" / "profiles"
    manifest = load_manifest(profile_dir / "qwen3_tts_0_6b_cuda.manifest.yaml")
    lock = load_lock(profile_dir / "qwen3_tts_0_6b_cuda.lock.json")
    manifest_models = {model.id: model for model in manifest.models}
    locked_models = {model.id: model for model in lock.models}
    assert manifest.profile_id == lock.profile_id == "qwen3_tts_0_6b_cuda"
    assert set(manifest_models) == set(locked_models)
    tts = locked_models["Qwen/Qwen3-TTS-12Hz-0.6B-Base"]
    assert tts.revision == "5d83992436eae1d760afd27aff78a71d676296fc"
    assert tts.license == "Apache-2.0"
    assert {file.path for file in tts.files} == {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    assert all(file.size > 0 and len(file.sha256) == 64 for file in tts.files)


@pytest.mark.parametrize(
    ("stem", "model_id", "revision", "filename"),
    [
        (
            "gemma4_26b_a4b",
            "google/gemma-4-26B-A4B-it",
            "3d3dca2094ff8112005fd10fc7a8e30cf4f45b56",
            "gemma-4-26B-A4B-it-Q4_0.gguf",
        ),
        (
            "qwen3_30b_a3b_2507",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "6c6e8692f43e4ca663f7ece8229a1361090d3a4c",
            "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        ),
    ],
)
def test_llm_candidate_artifacts_have_exact_matching_locks(
    stem: str,
    model_id: str,
    revision: str,
    filename: str,
) -> None:
    profile_dir = REPO_ROOT / "models" / "profiles"
    manifest = load_manifest(profile_dir / f"{stem}.manifest.yaml")
    lock = load_lock(profile_dir / f"{stem}.lock.json")
    assert manifest.profile_id == lock.profile_id == stem
    assert len(manifest.models) == len(lock.models) == 1
    artifact = lock.models[0]
    assert artifact.id == model_id
    assert artifact.revision == revision
    assert artifact.files[0].path == filename
    assert artifact.files[0].size > 10_000_000_000
    assert len(artifact.files[0].sha256) == 64


@pytest.mark.parametrize(
    ("stem", "llm_model"),
    [
        (
            "compat_gemma31_qwen17_ggml",
            "google/gemma-4-31B-it",
        ),
        (
            "multisession_gemma_a4b_qwen06_cuda",
            "google/gemma-4-26B-A4B-it",
        ),
        (
            "multisession_qwen_a3b_qwen06_cuda",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        ),
    ],
)
def test_startup_profiles_have_complete_matching_delivery_locks(
    stem: str,
    llm_model: str,
) -> None:
    profile_dir = REPO_ROOT / "models" / "profiles"
    manifest = load_manifest(profile_dir / f"{stem}.manifest.yaml")
    lock = load_lock(profile_dir / f"{stem}.lock.json")
    assert manifest.profile_id == lock.profile_id == stem
    assert {model.id for model in manifest.models} == {model.id for model in lock.models}
    expected_tts = (
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        if stem == "compat_gemma31_qwen17_ggml"
        else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    )
    expected_packages = (
        {"silero-vad"}
        if stem == "compat_gemma31_qwen17_ggml"
        else {"faster-qwen3-tts", "silero-vad"}
    )
    assert {model.id for model in lock.models} == {
        llm_model,
        "nvidia/parakeet-tdt-0.6b-v3",
        expected_tts,
        *expected_packages,
    }
    assert all(
        file.size > 0 and len(file.sha256) == 64 for model in lock.models for file in model.files
    )


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
        "profile_id": "test_profile",
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
            "profile_id": "test_profile",
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

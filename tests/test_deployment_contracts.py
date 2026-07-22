from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).parents[1]


def load_yaml(path: str) -> dict[str, Any]:
    value = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_gpu_service_installs_websocket_protocol() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "services/gpu-service/pyproject.toml").read_text(encoding="utf-8")
    )
    assert "websockets==15.0.1" in pyproject["project"]["dependencies"]


def test_kubernetes_gpu_and_security_contract() -> None:
    deployment = load_yaml("deploy/kubernetes/base/deployment.yaml")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    assert pod["automountServiceAccountToken"] is False
    assert pod["terminationGracePeriodSeconds"] >= 45
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    assert mounts["models"]["readOnly"] is True
    assert mounts["token"]["readOnly"] is True
    assert container["startupProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health/live"


def test_kustomization_never_installs_placeholder_secret_or_prefetch_job() -> None:
    kustomization = load_yaml("deploy/kubernetes/base/kustomization.yaml")
    resources = kustomization["resources"]
    assert "secret.yaml" not in resources
    assert "model-prefetch-job.yaml" not in resources
    assert (REPO_ROOT / "deploy/kubernetes/base/secret.yaml").is_file()
    assert (REPO_ROOT / "deploy/kubernetes/base/model-prefetch-job.yaml").is_file()


def test_prefetch_job_explicitly_enables_networked_model_delivery() -> None:
    job = load_yaml("deploy/kubernetes/base/model-prefetch-job.yaml")
    container = job["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item["value"] for item in container["env"]}
    assert environment["HF_HUB_OFFLINE"] == "0"
    assert environment["TRANSFORMERS_OFFLINE"] == "0"


def test_headless_discovery_and_no_autoscaling_objects() -> None:
    headless = load_yaml("deploy/kubernetes/base/headless-service.yaml")
    assert headless["spec"]["clusterIP"] == "None"
    assert headless["spec"]["publishNotReadyAddresses"] is False
    manifests = list((REPO_ROOT / "deploy/kubernetes").rglob("*.yaml"))
    kinds = {load_yaml(str(path.relative_to(REPO_ROOT))).get("kind") for path in manifests}
    assert "HorizontalPodAutoscaler" not in kinds
    assert "PodDisruptionBudget" not in kinds


def test_delivery_files_have_no_latest_or_embedded_model_copy() -> None:
    files = [
        REPO_ROOT / "services/gpu-service/Dockerfile",
        REPO_ROOT / "examples/minimal-livekit-agent/Dockerfile",
        *list((REPO_ROOT / "deploy").rglob("*.yaml")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert ":latest" not in text
    dockerfile = (REPO_ROOT / "services/gpu-service/Dockerfile").read_text(encoding="utf-8")
    assert "COPY .models" not in dockerfile
    assert "COPY models/manifest.lock.json" not in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "-DLLAMA_BUILD_UI=OFF" in dockerfile

    build_script = (REPO_ROOT / "services/gpu-service/scripts/build_llama_cpp.py").read_text(
        encoding="utf-8"
    )
    assert '"-DLLAMA_BUILD_UI=OFF"' in build_script


def test_compose_secret_is_readable_by_fixed_non_root_users() -> None:
    base = load_yaml("deploy/docker/compose.yaml")
    overlay = load_yaml("deploy/docker/compose.livekit.yaml")
    expected_group = ["${HUGGING_VOICE_SECRET_GID:-10001}"]
    service = base["services"]["hugging-voice"]
    agent = overlay["services"]["agent"]
    assert service["group_add"] == expected_group
    assert agent["group_add"] == expected_group
    assert "mode" not in service["secrets"][0]
    assert "mode" not in agent["secrets"][0]


def test_livekit_demo_exposes_web_ui_and_real_rtc_ports() -> None:
    services = load_yaml("deploy/docker/compose.livekit.yaml")["services"]
    livekit = services["livekit"]
    web = services["web"]
    assert "${LIVEKIT_BIND_ADDRESS:-127.0.0.1}:7880:7880" in livekit["ports"]
    assert "${LIVEKIT_RTC_BIND_ADDRESS:-0.0.0.0}:7881:7881" in livekit["ports"]
    assert "${LIVEKIT_RTC_BIND_ADDRESS:-0.0.0.0}:7882:7882/udp" in livekit["ports"]
    assert web["entrypoint"] == ["python", "/app/web.py"]
    assert web["ports"] == [
        "${HUGGING_VOICE_WEB_BIND_ADDRESS:-127.0.0.1}:${HUGGING_VOICE_WEB_PORT:-3000}:3000"
    ]
    assert web["environment"]["LIVEKIT_INTERNAL_URL"] == "ws://livekit:7880"


def test_compose_mounts_model_root_and_generated_lock_read_only() -> None:
    service = load_yaml("deploy/docker/compose.yaml")["services"]["hugging-voice"]
    mounts = {item["target"]: item for item in service["volumes"]}
    assert mounts["/models"]["source"] == "${HUGGING_VOICE_MODEL_ROOT:-../../.models}"
    assert mounts["/models"]["read_only"] is True
    lock_target = "/etc/hugging-voice/manifest.lock.json"
    assert mounts[lock_target]["source"] == (
        "${HUGGING_VOICE_MODEL_LOCK:-../../models/manifest.lock.json}"
    )
    assert mounts[lock_target]["read_only"] is True
    assert service["environment"]["HV_MODELS__LOCK_FILE"] == lock_target
    assert "/tmp:rw,noexec,nosuid,size=4g,uid=10001,gid=10001" in service["tmpfs"]

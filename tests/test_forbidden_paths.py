from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
PRODUCTION_ROOTS = (
    REPO_ROOT / "services/gpu-service/src",
    REPO_ROOT / "packages/livekit-plugins-hugging-voice/livekit",
    REPO_ROOT / "packages/hugging-voice-protocol/src",
    REPO_ROOT / "examples/minimal-livekit-agent",
)


def production_python() -> list[Path]:
    return [path for root in PRODUCTION_ROOTS for path in root.rglob("*.py")]


def test_production_has_no_forbidden_provider_or_runtime_paths() -> None:
    forbidden = re.compile(
        r"maistack|fastenhancer|deepfilternet|aiortc|webrtc|lightglue|\btrainer\b|"
        r"torch\.hub|torch\.jit\.script|from_pretrained|openai.{0,30}realtime|"
        r"realtime.{0,30}openai|\be4b\b",
        re.IGNORECASE,
    )
    findings = [
        f"{path.relative_to(REPO_ROOT)}:{line_number}:{line.strip()}"
        for path in production_python()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if forbidden.search(line)
        and not (
            path.name == "qwen_tts.py" and line.strip() == "model = FasterQwen3TTS.from_pretrained("
        )
    ]
    assert findings == []


def test_clients_cannot_carry_reference_audio() -> None:
    """Voice cloning is operator-defined only: the wire protocol, the LiveKit
    plugin, and the example worker must stay free of reference-audio inputs.
    (Naming the mode — e.g. the informational ``tts_mode`` field — is fine;
    carrying reference audio or transcripts is not.)"""

    client_roots = (
        REPO_ROOT / "packages/livekit-plugins-hugging-voice/livekit",
        REPO_ROOT / "packages/hugging-voice-protocol/src",
        REPO_ROOT / "examples/minimal-livekit-agent",
    )
    forbidden = re.compile(
        r"ref[_ -]?audio|ref[_ -]?text|reference[ _-]recording",
        re.IGNORECASE,
    )
    for root in client_roots:
        assert root.is_dir(), f"client root moved: {root}"
    findings = [
        f"{path.relative_to(REPO_ROOT)}:{line_number}:{line.strip()}"
        for root in client_roots
        for path in root.rglob("*.py")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if forbidden.search(line)
    ]
    assert findings == []


def test_mcp_and_cpu_fallback_mentions_are_fail_closed_only() -> None:
    mcp_lines: list[str] = []
    fallback_lines: list[str] = []
    for path in production_python():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "mcp" in line.lower():
                mcp_lines.append(line.strip())
            if "cpu fallback" in line.lower():
                fallback_lines.append(line.strip())
    assert mcp_lines == ['"--no-ui-mcp-proxy",']
    assert fallback_lines
    assert all("disabled" in line for line in fallback_lines)


def test_production_asyncio_queues_have_a_bound() -> None:
    findings: list[str] = []
    for path in production_python():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "asyncio"
                and node.func.attr in {"Queue", "LifoQueue", "PriorityQueue"}
                and not node.args
                and not any(keyword.arg == "maxsize" for keyword in node.keywords)
            ):
                findings.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert findings == []


def test_no_movable_latest_pins_in_delivery_files() -> None:
    paths = [
        REPO_ROOT / "services/gpu-service/Dockerfile",
        REPO_ROOT / "examples/minimal-livekit-agent/Dockerfile",
        *list((REPO_ROOT / "deploy").rglob("*.yaml")),
        *list((REPO_ROOT / ".github/workflows").rglob("*.yaml")),
    ]
    findings = [str(path.relative_to(REPO_ROOT)) for path in paths if ":latest" in path.read_text()]
    assert findings == []

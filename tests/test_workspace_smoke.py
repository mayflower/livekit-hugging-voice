from hugging_voice_protocol import PROTOCOL_VERSION
from hugging_voice_service import __version__ as service_version
from livekit.plugins import hugging_voice


def test_workspace_packages_import_together() -> None:
    assert PROTOCOL_VERSION == 2
    assert service_version == hugging_voice.__version__ == "0.3.0"

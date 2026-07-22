from hugging_voice_protocol import PROTOCOL_VERSION, WEBSOCKET_SUBPROTOCOL, __version__


def test_package_metadata_is_stable() -> None:
    assert __version__ == "0.1.0"
    assert PROTOCOL_VERSION == 1
    assert WEBSOCKET_SUBPROTOCOL == "hugging-voice-livekit.v1"

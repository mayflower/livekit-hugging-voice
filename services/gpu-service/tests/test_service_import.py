from hugging_voice_service import __version__


def test_service_package_imports() -> None:
    assert __version__ == "0.3.0"

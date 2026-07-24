from importlib import import_module


def test_plugin_namespace_imports() -> None:
    hugging_voice = import_module("livekit.plugins.hugging_voice")
    assert hugging_voice.__version__ == "0.3.0"

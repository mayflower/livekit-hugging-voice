from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def test_tts_quality_corpus_has_the_fixed_multilingual_coverage() -> None:
    path = Path("benchmarks/tts_quality_sentences.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    sentences = payload["sentences"]
    counts = Counter(item["language"] for item in sentences)
    assert counts == {"de": 20, "en": 10, "fr": 10, "it": 10}
    assert all(item["text"].strip() == item["text"] and item["text"] for item in sentences)
    assert len({(item["language"], item["text"]) for item in sentences}) == 50

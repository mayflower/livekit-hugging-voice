from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tool_eval_corpus",
    REPO_ROOT / "benchmarks" / "tool_eval_corpus.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CATEGORIES = MODULE.CATEGORIES
LANGUAGE_COUNTS = MODULE.LANGUAGE_COUNTS
TOOLS = MODULE.TOOLS
build_corpus = MODULE.build_corpus


def test_tool_eval_corpus_has_required_size_languages_and_categories() -> None:
    corpus = build_corpus()
    assert len(corpus) == 200
    assert Counter(case.language for case in corpus) == LANGUAGE_COUNTS
    assert set(case.category for case in corpus) == set(CATEGORIES)
    assert all(5 <= len(case.offered_tools) <= 10 for case in corpus)
    assert all(case.offered_tools == TOOLS for case in corpus)
    assert len({case.case_id for case in corpus}) == len(corpus)


def test_tool_eval_corpus_covers_choices_results_and_untrusted_output() -> None:
    corpus = build_corpus()
    assert {case.tool_choice for case in corpus} == {"auto", "required", "none", "named"}
    assert all(case.named_tool in TOOLS for case in corpus if case.tool_choice == "named")
    assert any(case.category == "sequential_two_step_flow" for case in corpus)
    assert any(case.category == "tool_error_result" for case in corpus)
    injection = [case for case in corpus if case.category == "prompt_injection_like_tool_output"]
    assert injection
    assert all(case.expect_result_in_final for case in injection)


def test_tool_eval_ground_truth_matches_each_semantic_category() -> None:
    corpus = build_corpus()
    expected_names = {
        "auto_tool": "lookup_weather",
        "required_tool": "add_numbers",
        "named_tool": "find_contact",
        "optional_arguments": "lookup_weather",
        "enums": "convert_currency",
        "numbers_and_dates": "create_calendar_event",
        "similar_tool_names": "lookup_order",
        "sequential_two_step_flow": "lookup_order",
        "tool_error_result": "lookup_weather",
        "prompt_injection_like_tool_output": "find_contact",
    }
    for case in corpus:
        if case.category in {"no_tool", "tool_not_applicable"}:
            assert case.tool_choice == ("none" if case.category == "no_tool" else "auto")
            assert case.expected_tool is None
            assert case.expected_arguments is None
            continue
        assert case.expected_tool == expected_names[case.category]
        assert case.expected_arguments
        assert case.expected_tool in case.offered_tools
    assert "lookup_order_status" in TOOLS

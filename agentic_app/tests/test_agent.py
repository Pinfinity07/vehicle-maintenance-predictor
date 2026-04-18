"""Tests for agent.py — negation handling, intent routing, and collection flow."""

from unittest.mock import patch, MagicMock

import pytest

from agent import (
    apply_negation_heuristics,
    classify_intent,
    answer_question,
    get_missing_features,
    handle_message,
    REQUIRED_FEATURES,
)


# ---------------------------------------------------------------------------
# Negation heuristics (pure, no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I don't have insurance",
    "no insurance on this vehicle",
    "the car is uninsured",
    "I drive without insurance",
    "there's no insurance premium",
])
def test_negation_insurance(text):
    result = apply_negation_heuristics(text, collected={})
    assert result.get("Insurance_Premium") == 0, f"failed for: {text}"


@pytest.mark.parametrize("text", [
    "no accidents",
    "never had an accident",
    "it's accident-free",
    "zero accidents",
    "never been in an accident",
])
def test_negation_accidents(text):
    result = apply_negation_heuristics(text, collected={})
    assert result.get("Accident_History") == 0, f"failed for: {text}"


@pytest.mark.parametrize("text", [
    "no issues",
    "no reported issues",
    "no problems",
    "no complaints",
])
def test_negation_issues(text):
    result = apply_negation_heuristics(text, collected={})
    assert result.get("Reported_Issues") == 0, f"failed for: {text}"


def test_negation_never_serviced():
    assert apply_negation_heuristics("never serviced", {})["Service_History"] == 0
    assert apply_negation_heuristics("no service history", {})["Service_History"] == 0


def test_negation_does_not_overwrite_collected():
    result = apply_negation_heuristics(
        "no insurance", collected={"Insurance_Premium": 12000}
    )
    assert "Insurance_Premium" not in result


def test_negation_ignores_positive_mentions():
    # "I have insurance" should NOT set Insurance_Premium=0
    result = apply_negation_heuristics("I have insurance worth 12000", {})
    assert "Insurance_Premium" not in result


def test_negation_multiple_features():
    result = apply_negation_heuristics(
        "no insurance, no accidents, no reported issues", {}
    )
    assert result["Insurance_Premium"] == 0
    assert result["Accident_History"] == 0
    assert result["Reported_Issues"] == 0


# ---------------------------------------------------------------------------
# Missing-features logic
# ---------------------------------------------------------------------------

def test_get_missing_features_empty():
    missing = get_missing_features({})
    assert len(missing) == len(REQUIRED_FEATURES)


def test_get_missing_features_with_zero():
    # Critical: zero values (like no insurance) should count as collected
    collected = {"Insurance_Premium": 0, "Accident_History": 0}
    missing = get_missing_features(collected)
    assert "Insurance_Premium" not in missing
    assert "Accident_History" not in missing


# ---------------------------------------------------------------------------
# Intent classification (mocked LLM)
# ---------------------------------------------------------------------------

def _mock_llm_returning(text: str):
    """Helper to build a mocked ChatGroq-like object."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=text)
    return mock


def test_classify_intent_question():
    with patch("agent._get_llm") as mock_get:
        mock_get.return_value = _mock_llm_returning(
            '{"intent": "question", "topic": "owner types"}'
        )
        result = classify_intent("what are the owner types", api_key="fake")
        assert result["intent"] == "question"


def test_classify_intent_info():
    with patch("agent._get_llm") as mock_get:
        mock_get.return_value = _mock_llm_returning(
            '{"intent": "info", "topic": ""}'
        )
        result = classify_intent("I have a 5 year old truck", api_key="fake")
        assert result["intent"] == "info"


def test_classify_intent_fallback_when_no_llm():
    with patch("agent._get_llm") as mock_get:
        mock_get.return_value = None
        result = classify_intent("anything", api_key=None)
        assert result["intent"] == "info"


# ---------------------------------------------------------------------------
# answer_question without LLM (fallback path)
# ---------------------------------------------------------------------------

def test_answer_question_fallback_lists_options():
    with patch("agent._get_llm") as mock_get:
        mock_get.return_value = None
        answer = answer_question(
            "what are the owner types",
            missing=["Owner_Type", "Fuel_Type"],
            collected={},
            api_key=None,
        )
        assert "Owner" in answer or "owner" in answer.lower()
        assert "First" in answer
        assert "Second" in answer
        assert "Third" in answer


# ---------------------------------------------------------------------------
# handle_message integration (heavily mocked)
# ---------------------------------------------------------------------------

def test_handle_message_question_does_not_extract():
    """When user asks a question, bot should NOT try to extract features,
    and should answer instead."""
    with patch("agent._get_llm") as mock_get:
        # 1st call: intent classifier -> question
        # 2nd call: answer_question -> direct answer
        calls = iter([
            _mock_llm_returning('{"intent": "question", "topic": "owner"}'),
            _mock_llm_returning("Owner Type is one of: First, Second, or Third."),
        ])
        mock_get.side_effect = lambda *a, **kw: next(calls)

        response, features, extra, phase = handle_message(
            "what are the owner types?",
            chat_history=[],
            collected_features={},
            extra_info=[],
            api_key="fake",
        )
        assert phase == "collecting"
        assert features == {}  # nothing extracted
        assert "First" in response


def test_handle_message_negation_records_zero_insurance():
    """User saying 'I have no insurance' should set Insurance_Premium=0 even
    if the LLM extractor misses it."""
    with patch("agent._get_llm") as mock_get:
        # intent -> info; extractor returns empty (LLM missed the negation)
        calls = iter([
            _mock_llm_returning('{"intent": "info", "topic": ""}'),
            _mock_llm_returning('{"extracted": {}, "extra_info": [], "confidence": "high"}'),
        ])
        mock_get.side_effect = lambda *a, **kw: next(calls)

        response, features, extra, phase = handle_message(
            "I don't have any insurance",
            chat_history=[],
            collected_features={},
            extra_info=[],
            api_key="fake",
        )
        # Deterministic negation heuristic must have caught it
        assert features.get("Insurance_Premium") == 0


def test_handle_message_subsequent_turn_does_not_reask_insurance():
    """After insurance is set to 0 via negation, next turn should not ask
    for insurance again."""
    with patch("agent._get_llm") as mock_get:
        calls = iter([
            _mock_llm_returning('{"intent": "info", "topic": ""}'),
            _mock_llm_returning(
                '{"extracted": {"Vehicle_Age": 5}, "extra_info": [], "confidence": "high"}'
            ),
        ])
        mock_get.side_effect = lambda *a, **kw: next(calls)

        collected = {"Insurance_Premium": 0}
        response, features, extra, phase = handle_message(
            "it's 5 years old",
            chat_history=[],
            collected_features=collected,
            extra_info=[],
            api_key="fake",
        )
        # Insurance_Premium should still be 0 and not re-asked
        assert features.get("Insurance_Premium") == 0
        assert "insurance premium" not in response.lower()

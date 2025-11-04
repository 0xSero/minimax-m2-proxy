"""Unit tests for reasoning helpers."""

from parsers.reasoning import ensure_think_wrapped, split_think
from proxy.session_store import SessionStore


def test_split_think_separates_reasoning_from_visible() -> None:
    reasoning, visible = split_think(
        "<think>Evaluate options</think>\nProvide concise response."
    )

    assert reasoning.strip() == "Evaluate options"
    assert visible.strip() == "Provide concise response."


def test_split_think_handles_missing_opening_tag() -> None:
    reasoning, visible = split_think("Working backwards</think>\nResult summary.")

    assert reasoning.strip() == "Working backwards"
    assert visible.strip() == "Result summary."


def test_ensure_think_wrapped_adds_missing_opening_tag() -> None:
    wrapped = ensure_think_wrapped("analysis</think>\nAnswer")
    assert wrapped.startswith("<think>")
    assert wrapped.count("<think>") == wrapped.count("</think>")


def test_session_store_normalizes_assistant_reasoning() -> None:
    message = {
        "role": "assistant",
        "content": "Visible answer",
        "reasoning_details": [{"type": "chain_of_thought", "text": "Internal reasoning"}],
    }

    normalized = SessionStore._normalize_assistant_message(message)
    assert normalized["content"].startswith("<think>Internal reasoning</think>")
    assert "Visible answer" in normalized["content"]
    assert "reasoning_details" not in normalized

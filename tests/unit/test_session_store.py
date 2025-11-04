"""Unit tests for the session store repair helpers."""

from proxy.session_store import SessionStore


def test_inject_or_repair_inserts_missing_assistant_turn() -> None:
    store = SessionStore(
        enabled=True,
        backend="memory",
        db_path=":memory:",
        ttl_seconds=60,
        max_messages=8,
    )

    session_id = "abc"
    assistant_message = {"role": "assistant", "content": "<think>r</think>\nHello"}
    store.append_message(session_id, assistant_message)

    user_turn = [{"role": "user", "content": "Hi"}]
    result = store.inject_or_repair(user_turn, session_id, require_session=True)

    assert result.repaired is True
    assert result.messages[0] == {"role": "user", "content": "Hi"}
    assert result.messages[1] == assistant_message


def test_inject_or_repair_skips_when_assistant_present() -> None:
    store = SessionStore(
        enabled=True,
        backend="memory",
        db_path=":memory:",
        ttl_seconds=60,
        max_messages=8,
    )

    session_id = "abc"
    assistant_message = {"role": "assistant", "content": "<think>r</think>\nHello"}
    store.append_message(session_id, assistant_message)

    history = [
        {"role": "assistant", "content": "<think>r</think>\nHello"},
        {"role": "user", "content": "Tool output"},
    ]
    result = store.inject_or_repair(history, session_id, require_session=True)

    assert result.repaired is False
    assert result.skipped is True
    assert result.skip_reason == "assistant_present"

"""Shared pytest fixtures for the MiniMax-M2 proxy."""

from __future__ import annotations

from collections import deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional

import pytest

from proxy import main as proxy_main
from proxy.session_store import session_store


class StubTabbyClient:
    """Deterministic stub for TabbyAPI interactions."""

    def __init__(self) -> None:
        self.chat_responses: Deque[Dict[str, Any]] = deque()
        self.streaming_sequences: Deque[List[Dict[str, Any]]] = deque()
        self.calls: List[Dict[str, Any]] = []

    def queue_chat_response(self, response: Dict[str, Any]) -> None:
        self.chat_responses.append(response)

    def queue_stream(self, chunks: List[Dict[str, Any]]) -> None:
        self.streaming_sequences.append(chunks)

    async def chat_completion(self, *_, **kwargs: Any) -> Dict[str, Any]:
        if not self.chat_responses:
            raise RuntimeError("StubTabbyClient.chat_responses is empty")
        self.calls.append({"type": "chat", "payload": kwargs})
        return self.chat_responses.popleft()

    async def extract_streaming_content(
        self,
        *_: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self.streaming_sequences:
            raise RuntimeError("StubTabbyClient.streaming_sequences is empty")
        self.calls.append({"type": "stream", "payload": kwargs})
        for chunk in self.streaming_sequences.popleft():
            yield chunk

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.fixture(name="stub_tabby")
def fixture_stub_tabby(monkeypatch: pytest.MonkeyPatch) -> StubTabbyClient:
    """Provide a stub Tabby client and install it into the proxy module."""
    stub = StubTabbyClient()
    original_client = proxy_main.tabby_client
    proxy_main.tabby_client = stub
    try:
        yield stub
    finally:
        proxy_main.tabby_client = original_client


@pytest.fixture(autouse=True)
def reset_session_store() -> None:
    """Ensure the session store is empty between tests."""
    if hasattr(session_store, "_memory_store"):
        session_store._memory_store.clear()  # type: ignore[attr-defined]

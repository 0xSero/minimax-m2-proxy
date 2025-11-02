"""Utility helpers for MiniMax-M2 reasoning (<think>) handling."""

from __future__ import annotations

from typing import Tuple


def ensure_think_wrapped(text: str) -> Tuple[str, bool]:
    """Ensure completions that contain ``</think>`` start with an opening tag.

    TabbyAPI omits the opening ``<think>`` because it is part of the prompt.
    We add it back exactly once when a closing tag is present and an opening
    tag is missing. The helper preserves any leading whitespace that may carry
    streaming formatting semantics.

    Returns the possibly modified text and a flag indicating whether an
    opening tag was inserted.
    """

    if not text or "</think>" not in text:
        return text, False

    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        return text, False

    leading_len = len(text) - len(stripped)
    leading_ws = text[:leading_len]

    # Insert a newline after <think> unless one is already present so the
    # reasoning block keeps the expected shape.
    remainder = stripped
    newline = "" if remainder.startswith("\n") else "\n"
    return f"{leading_ws}<think>{newline}{remainder}", True

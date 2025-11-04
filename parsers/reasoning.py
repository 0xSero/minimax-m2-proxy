"""Utility for MiniMax-M2 reasoning (<think>) tag handling"""

from __future__ import annotations

from typing import Tuple


def ensure_think_wrapped(text: str) -> str:
    """
    Add missing <think> opening tag if </think> is present.

    TabbyAPI omits the opening <think> tag because it's in the prompt.
    This helper adds it back when needed.
    """
    if not text or "</think>" not in text:
        return text

    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        return text

    # Add opening tag and preserve original indentation/newlines on first line
    prefix_len = len(text) - len(stripped)
    prefix = text[:prefix_len]
    return f"{prefix}<think>\n{stripped}"


def split_think(text: str) -> Tuple[str, str]:
    """
    Split thinking content (inside <think>...</think>) from visible content.

    Returns:
        (reasoning_text, visible_text)
    """
    if not text:
        return "", ""

    reasoning_parts: list[str] = []
    visible_parts: list[str] = []
    remaining = text

    while remaining:
        open_idx = remaining.find("<think>")
        close_idx = remaining.find("</think>")

        if open_idx == -1 and close_idx == -1:
            visible_parts.append(remaining)
            break

        if 0 <= open_idx < (close_idx if close_idx != -1 else len(remaining)):
            # Visible text before <think>
            if open_idx > 0:
                visible_parts.append(remaining[:open_idx])
            remaining = remaining[open_idx + len("<think>") :]
            close_idx = remaining.find("</think>")
            if close_idx == -1:
                reasoning_parts.append(remaining)
                break
            reasoning_parts.append(remaining[:close_idx])
            remaining = remaining[close_idx + len("</think>") :]
        elif close_idx != -1:
            # Closing tag without explicit opening in this segment (template implicit opening)
            reasoning_parts.append(remaining[:close_idx])
            remaining = remaining[close_idx + len("</think>") :]
        else:
            visible_parts.append(remaining)
            break

    return "".join(reasoning_parts), "".join(visible_parts)

"""Utility for MiniMax-M2 reasoning (<think>) tag handling"""


def ensure_think_wrapped(text: str) -> str:
    """
    Add missing <think> opening tag if </think> is present.

    TabbyAPI omits the opening <think> tag because it's in the prompt.
    This helper adds it back when needed.
    """
    if not text or "</think>" not in text:
        return text

    if text.strip().startswith("<think>"):
        return text

    # Add opening tag
    return f"<think>\n{text.lstrip()}"

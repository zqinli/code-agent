"""Protocol parser for code-agent model actions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedAction:
    action_type: str
    content: str
    raw_text: str
    think: str = ""


ACTION_TAGS = ("search_code", "open_file", "run_sandbox", "generate_patch", "final")
ACTION_TAG_RE = "|".join(ACTION_TAGS)


def _strip(text: str | None) -> str:
    return "" if text is None else str(text).strip()


def parse_action(text: str | None) -> ParsedAction:
    """Parse exactly one model action from raw model output."""
    text = "" if text is None else str(text)

    think_action = re.compile(
        r"(?P<raw>\s*<think>(?P<think>.*?)</think>\s*"
        rf"<(?P<tag>{ACTION_TAG_RE})>(?P<content>.*?)</(?P=tag)>"
        r"(?:\s*<final>.*?</final>)?)",
        flags=re.DOTALL,
    )
    match = think_action.search(text)
    if match:
        return ParsedAction(
            action_type=match.group("tag").strip(),
            content=match.group("content").strip(),
            raw_text=match.group("raw").strip(),
            think=match.group("think").strip(),
        )

    direct_action = re.compile(
        rf"(?P<raw>\s*<(?P<tag>{ACTION_TAG_RE})>(?P<content>.*?)</(?P=tag)>"
        r"(?:\s*<final>.*?</final>)?)",
        flags=re.DOTALL,
    )
    match = direct_action.search(text)
    if match:
        return ParsedAction(
            action_type=match.group("tag").strip(),
            content=match.group("content").strip(),
            raw_text=match.group("raw").strip(),
        )

    invalid_raw = _strip(text)
    if len(invalid_raw) > 2000:
        invalid_raw = invalid_raw[:2000]
    return ParsedAction(action_type="invalid", content=invalid_raw, raw_text=invalid_raw)


def parse_final_action(text: str | None) -> ParsedAction:
    """Prefer patch/final actions, then fall back to normal action parsing."""
    text = "" if text is None else str(text)

    think_terminal = re.compile(
        r"(?P<raw>\s*<think>(?P<think>.*?)</think>\s*"
        r"<(?P<tag>generate_patch|final)>(?P<content>.*?)</(?P=tag)>"
        r"(?:\s*<final>.*?</final>)?)",
        flags=re.DOTALL,
    )
    match = think_terminal.search(text)
    if match:
        return ParsedAction(
            action_type=match.group("tag").strip(),
            content=match.group("content").strip(),
            raw_text=match.group("raw").strip(),
            think=match.group("think").strip(),
        )

    direct_terminal = re.compile(
        r"(?P<raw>\s*<(?P<tag>generate_patch|final)>(?P<content>.*?)</(?P=tag)>"
        r"(?:\s*<final>.*?</final>)?)",
        flags=re.DOTALL,
    )
    match = direct_terminal.search(text)
    if match:
        return ParsedAction(
            action_type=match.group("tag").strip(),
            content=match.group("content").strip(),
            raw_text=match.group("raw").strip(),
        )

    return parse_action(text)

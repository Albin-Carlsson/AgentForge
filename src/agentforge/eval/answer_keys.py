"""Deterministic answer correctness: does the answer state the key facts?

Each frozen question carries `answer_keys`, the facts a correct answer must
contain, with `|` separating accepted spellings of one fact. The score is the
fraction of keys found. No model is involved, so it costs nothing, has no
variance, and needs no validation -- which is exactly why it is used instead
of a correctness judge that would itself need a kappa.

The cost is strictness: "seventy-six percent" does not match "76%". That bias
is identical across arms, so it shifts every arm's level but not the deltas.
"""

from __future__ import annotations

import re

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_SPACE_PERCENT = re.compile(r"(?<=\d)\s+%")


def normalize(text: str) -> str:
    text = text.lower().replace("−", "-").replace("–", "-").replace("‑", "-")
    text = _THOUSANDS.sub("", text)
    text = _SPACE_PERCENT.sub("%", text)
    return re.sub(r"\s+", " ", text)


def key_found(answer: str, key: str) -> bool:
    haystack = normalize(answer)
    return any(normalize(alt) in haystack for alt in key.split("|"))


def key_recall(answer: str, keys: list[str]) -> float:
    if not keys:
        raise ValueError("a question with no answer keys cannot be scored")
    return sum(key_found(answer, k) for k in keys) / len(keys)

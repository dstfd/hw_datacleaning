"""Shared normalization + fuzzy helpers."""

from __future__ import annotations
import re
import pandas as pd
from rapidfuzz import fuzz, process


def norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def best_match(query: str, choices: list[str], scorer=fuzz.token_sort_ratio):
    """Returns (match_string, score, index) or None."""
    return process.extractOne(query, choices, scorer=scorer)


def top_k_matches(
    query: str, choices: list[str], k: int = 5, scorer=fuzz.token_sort_ratio
) -> list[tuple[str, float, int]]:
    """Returns top-k matches as [(name, score, original_index), ...] descending."""
    return list(process.extract(query, choices, scorer=scorer, limit=k))

"""
features.py — Pairwise feature engineering for entity resolution.

All computations are pure-Python / numpy (no Levenshtein C-extension required)
so they work in any environment.
"""
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Abbreviation expansion maps
# ---------------------------------------------------------------------------
NAME_ABBREVS = {
    r'\bcorp\b': 'corporation', r'\binc\b': 'incorporated', r'\bltd\b': 'limited',
    r'\bllc\b': 'limited liability company', r'\bllp\b': 'limited liability partnership',
    r'\bpvt\b': 'private', r'\bco\b': 'company', r'\bbros\b': 'brothers',
    r'\bintl\b': 'international', r'\bmfg\b': 'manufacturing',
    r'\bsvc\b': 'services', r'\bsvcs\b': 'services', r'\bgrp\b': 'group',
    r'\basso\b': 'associates', r'\bassoc\b': 'associates',
    r'\bdba\b': '', r'\bthe\b': '',
    r'&': 'and',
}

ADDR_ABBREVS = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bavenue\b': 'avenue', r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive',
    r'\bln\b': 'lane', r'\bct\b': 'court', r'\bpl\b': 'place',
    r'\bsq\b': 'square', r'\bpkwy\b': 'parkway', r'\bfwy\b': 'freeway',
    r'\bhwy\b': 'highway', r'\bexpy\b': 'expressway',
    r'\bste\b': 'suite', r'\bapt\b': 'apartment', r'\bfl\b': 'floor',
    r'\bn\b': 'north', r'\bs\b': 'south', r'\be\b': 'east', r'\bw\b': 'west',
    r'\bne\b': 'northeast', r'\bnw\b': 'northwest', r'\bse\b': 'southeast', r'\bsw\b': 'southwest',
}

def _compile_map(abbrev_map: dict) -> list[tuple]:
    return [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in abbrev_map.items()]

_NAME_PATTERNS = _compile_map(NAME_ABBREVS)
_ADDR_PATTERNS = _compile_map(ADDR_ABBREVS)


def normalize_text(text: str, patterns: list[tuple]) -> str:
    """Lowercase, unicode-normalize, expand abbreviations, strip punctuation."""
    if not text or not isinstance(text, str):
        return ''
    # Unicode normalize (handles accents etc.)
    text = unicodedata.normalize('NFKC', text).lower().strip()
    # Expand abbreviations
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    # Remove punctuation except alphanumeric and space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(text: str) -> str:
    return normalize_text(text, _NAME_PATTERNS)


def normalize_address(text: str) -> str:
    return normalize_text(text, _ADDR_PATTERNS)


def combined_text(name: str, address: str) -> str:
    """Concatenated normalized name + address for TF-IDF indexing."""
    return f"{normalize_name(name)} {normalize_address(address)}"


# ---------------------------------------------------------------------------
# String similarity helpers
# ---------------------------------------------------------------------------

def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def char_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a[i:i+n] for i in range(len(a) - n + 1)) if len(a) >= n else set(a)
    sb = set(b[i:i+n] for i in range(len(b) - n + 1)) if len(b) >= n else set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def seq_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio (approximation of edit similarity)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def token_sort_ratio(a: str, b: str) -> float:
    """Token-sorted sequence ratio (handles word reordering)."""
    a_sorted = ' '.join(sorted(a.split()))
    b_sorted = ' '.join(sorted(b.split()))
    return seq_ratio(a_sorted, b_sorted)


# ---------------------------------------------------------------------------
# Feature vector for a candidate pair
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'name_token_jaccard',
    'name_char3_jaccard',
    'name_seq_ratio',
    'name_token_sort_ratio',
    'addr_token_jaccard',
    'addr_char3_jaccard',
    'addr_seq_ratio',
    'country_match',
    'tfidf_cosine',         # filled in by caller from blocking scores
]


def compute_features(
    s1_name_norm: str, s1_addr_norm: str, s1_country: str,
    sx_name_norm: str, sx_addr_norm: str, sx_country: str,
    tfidf_cosine: float = 0.0,
) -> list[float]:
    """Return a feature vector for a single candidate pair."""
    return [
        token_jaccard(s1_name_norm, sx_name_norm),
        char_jaccard(s1_name_norm, sx_name_norm, 3),
        seq_ratio(s1_name_norm, sx_name_norm),
        token_sort_ratio(s1_name_norm, sx_name_norm),
        token_jaccard(s1_addr_norm, sx_addr_norm),
        char_jaccard(s1_addr_norm, sx_addr_norm, 3),
        seq_ratio(s1_addr_norm, sx_addr_norm),
        float(s1_country.strip().lower() == sx_country.strip().lower()),
        float(tfidf_cosine),
    ]

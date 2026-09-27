"""
Feature engineering module for pairwise business entity matching.
Extracts 16 highly discriminative, computationally efficient features across
name similarity, address similarity, numeric tokens, country compatibility, and blocking provenance.
Includes vectorized batch extraction for high-throughput pipeline execution.
"""

from typing import Set, Tuple, List
import numpy as np

try:
    from src.candidate_scoring import jaccard_similarity
except ImportError:
    from .candidate_scoring import jaccard_similarity


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def char_dice_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Compute character n-gram Dice similarity coefficient."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)
    total = len(ngrams1) + len(ngrams2)
    if total == 0:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    return (2.0 * intersection) / total


# Standard set Jaccard similarity
jaccard = jaccard_similarity


FEATURE_NAMES = [
    "exact_name_match",
    "name_char_dice",
    "name_len_diff",
    "name_len_ratio",
    "name_token_jaccard",
    "first_name_token_match",
    "addr_char_dice",
    "addr_token_jaccard",
    "both_addr_empty",
    "one_addr_empty",
    "country_match",
    "is_s2",
    "is_s3",
    "rule_count",
    "has_exact_rule",
    "cheap_score",
]


def extract_features_for_pair(
    s1_name: str, s1_nc: str, s1_lc: str, s1_addr: str, s1_ac: str, s1_c: str,
    c_name: str, c_nc: str, c_lc: str, c_addr: str, c_ac: str, c_c: str,
    cand_id: str, matched_rules: int, has_exact_rule: int
) -> List[float]:
    """Extract 16 features for a candidate pair."""
    # 1. Exact Name Match
    exact_name = 1.0 if (s1_lc and s1_lc == c_lc) or (s1_nc and s1_nc == c_nc) else 0.0

    # 2. Name Dice
    name_dice = char_dice_similarity(s1_lc, c_lc, n=3)

    # 3. Name length features
    len_s1 = len(s1_lc)
    len_c = len(c_lc)
    max_len = max(len_s1, len_c)
    len_diff = float(abs(len_s1 - len_c))
    len_ratio = (min(len_s1, len_c) / max_len) if max_len > 0 else 1.0

    # 4. Name Token Jaccard
    s1_toks = set(s1_name.lower().split()) if s1_name else set()
    c_toks = set(c_name.lower().split()) if c_name else set()
    name_tok_jaccard = jaccard(s1_toks, c_toks)

    # 5. First Token Match
    s1_first = s1_name.split()[0].lower() if s1_name else ""
    c_first = c_name.split()[0].lower() if c_name else ""
    first_match = 1.0 if (s1_first and s1_first == c_first) else 0.0

    # 6. Address Features
    addr_dice = char_dice_similarity(s1_ac, c_ac, n=3)
    s1_a_toks = set(s1_addr.lower().split()) if s1_addr else set()
    c_a_toks = set(c_addr.lower().split()) if c_addr else set()
    addr_tok_jaccard = jaccard(s1_a_toks, c_a_toks)

    s1_empty = (len(s1_ac) == 0)
    c_empty = (len(c_ac) == 0)
    both_empty = 1.0 if (s1_empty and c_empty) else 0.0
    one_empty = 1.0 if (s1_empty != c_empty) else 0.0

    # 7. Country & Source
    country_match = 1.0 if (s1_c and s1_c == c_c) else 0.0
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    # 8. Provenance
    rc = float(matched_rules)
    exact_rule = float(has_exact_rule)
    cheap = exact_rule * 0.5 + name_dice * 0.3 + addr_dice * 0.2

    return [
        exact_name,
        name_dice,
        len_diff,
        len_ratio,
        name_tok_jaccard,
        first_match,
        addr_dice,
        addr_tok_jaccard,
        both_empty,
        one_empty,
        country_match,
        is_s2,
        is_s3,
        rc,
        exact_rule,
        cheap,
    ]


def extract_features_batch(rows: List[Tuple]) -> np.ndarray:
    """
    Vectorized batch feature extraction from DuckDB candidate rows.
    Row schema expected:
    (s1_id, cand_id, matched_rules, has_exact_rule,
     s1_name, s1_nc, s1_lc, s1_addr, s1_ac, s1_c,
     c_name, c_nc, c_lc, c_addr, c_ac, c_c)
    Returns: float32 ndarray of shape (N, 16)
    """
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    feats = []
    for r in rows:
        cand_id = r[1]
        matched_rules = r[2]
        has_exact_rule = r[3]
        s1_name = r[4] or ""
        s1_nc = r[5] or ""
        s1_lc = r[6] or ""
        s1_addr = r[7] or ""
        s1_ac = r[8] or ""
        s1_c = r[9] or ""
        c_name = r[10] or ""
        c_nc = r[11] or ""
        c_lc = r[12] or ""
        c_addr = r[13] or ""
        c_ac = r[14] or ""
        c_c = r[15] or ""

        f = extract_features_for_pair(
            s1_name, s1_nc, s1_lc, s1_addr, s1_ac, s1_c,
            c_name, c_nc, c_lc, c_addr, c_ac, c_c,
            cand_id, matched_rules, has_exact_rule
        )
        feats.append(f)

    return np.array(feats, dtype=np.float32)


# Backwards compatibility alias
extract_pairwise_features = extract_features_for_pair

"""
Deterministic cheap candidate scoring module.
Computes inexpensive heuristic scores to rank and prune candidate pools
before expensive ML feature engineering.
"""

from typing import Set, List, Dict, Tuple, Optional

# Weights for blocking rules reflecting their discriminative precision
RULE_WEIGHTS = {
    "exact_name": 1.2,
    "name_country": 1.0,
    "latin_name": 0.9,
    "latin_country": 0.8,
    "name_p6": 0.5,
    "latin_p6": 0.4,
    "name_tok0": 0.4,
    "name_tok1": 0.3,
    "addr_p10": 0.5,
    "addr_num_word": 0.5,
    "addr_num_word2": 0.3,
}


def jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard token similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def compute_cheap_score(
    s1_norm: Tuple[str, str, str, str, str, Set[str], Set[str], Set[str]],
    cand_norm: Tuple[str, str, str, str, str, Set[str], Set[str], Set[str]],
    matched_rules: Set[str]
) -> float:
    """
    Compute a fast deterministic score combining rule provenance and token overlap.
    s1_norm / cand_norm tuple schema:
    (name_norm, name_compact, addr_norm, addr_compact, country, name_toks, addr_toks, num_toks)
    """
    (
        s1_name_norm, s1_name_compact, s1_addr_norm, s1_addr_compact,
        s1_country, s1_name_toks, s1_addr_toks, s1_num_toks
    ) = s1_norm

    (
        c_name_norm, c_name_compact, c_addr_norm, c_addr_compact,
        c_country, c_name_toks, c_addr_toks, c_num_toks
    ) = cand_norm

    # Base rule score
    rule_score = sum(RULE_WEIGHTS.get(r, 0.2) for r in matched_rules)

    # Name similarity
    name_jaccard = jaccard_similarity(s1_name_toks, c_name_toks)
    exact_name_bonus = 1.0 if s1_name_compact and s1_name_compact == c_name_compact else 0.0

    # Address numeric similarity
    num_common = len(s1_num_toks & c_num_toks)
    num_bonus = 0.5 if num_common > 0 else 0.0

    # Address token similarity
    addr_jaccard = jaccard_similarity(s1_addr_toks, c_addr_toks)

    # Country compatibility
    country_penalty = 0.0 if not s1_country or not c_country or s1_country == c_country else -2.0

    total_score = (
        rule_score +
        exact_name_bonus +
        2.0 * name_jaccard +
        num_bonus +
        1.0 * addr_jaccard +
        country_penalty
    )

    return total_score

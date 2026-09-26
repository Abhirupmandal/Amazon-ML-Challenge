"""
Deterministic normalization and multilingual text handling module.
Preserves native Unicode representations while deriving clean transliterated,
compact, tokenized, and numeric-token representations for indexing and blocking.
"""

import re
import unicodedata
from typing import List, Tuple, Set, Optional

try:
    from unidecode import unidecode
except ImportError:
    # Fallback if unidecode is not present
    def unidecode(text: str) -> str:
        return text

# Comprehensive legal suffixes sorted by descending token length
LEGAL_SUFFIXES = [
    "private limited", "private ltd", "pvt limited", "pvt ltd",
    "limited liability company", "limited liability partnership",
    "joint stock company", "public limited company",
    "corporation", "incorporated", "company", "limited", "corp", "inc",
    "co", "ltd", "llc", "llp", "plc", "gmbh", "sarl", "sa", "ag",
    "lp", "pc", "holding", "holdings", "group", "enterprises"
]

RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")


def normalize_unicode(text: Optional[str]) -> str:
    """Normalize unicode with NFKC, casefold, replace '&', and strip punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.replace("&", " and ")
    text = RE_PUNCT.sub(" ", text)
    return " ".join(text.split())


def remove_legal_suffix(norm_text: str) -> str:
    """Strip standard legal entity suffixes from the end of a normalized name."""
    if not norm_text:
        return ""
    for suffix in LEGAL_SUFFIXES:
        if norm_text.endswith(" " + suffix):
            return norm_text[:-len(suffix) - 1].strip()
        elif norm_text == suffix:
            return ""
    return norm_text


def normalize_name(name: Optional[str]) -> str:
    """Normalize business name: unicode cleaning, lowercasing, legal suffix removal."""
    return remove_legal_suffix(normalize_unicode(name))


def latinize_name(name: Optional[str]) -> str:
    """
    Produce a Latin transliteration of a business name for cross-script matching.
    Only invokes unidecode when non-ASCII characters are present.
    """
    cleaned = normalize_name(name)
    if not cleaned or cleaned.isascii():
        return cleaned
    try:
        return normalize_name(unidecode(cleaned))
    except Exception:
        return cleaned


def compact(text: Optional[str]) -> str:
    """Remove all whitespace characters from text."""
    if not text:
        return ""
    return RE_WHITESPACE.sub("", text)


def normalize_address(address: Optional[str]) -> str:
    """Normalize business address: unicode cleaning and whitespace consolidation."""
    return normalize_unicode(address)


def normalize_country(country: Optional[str]) -> str:
    """Clean country code or name as an open string without hardcoding."""
    if not country:
        return ""
    return str(country).strip().upper()


def extract_tokens(text: Optional[str], min_length: int = 2) -> List[str]:
    """Extract list of words/tokens with length >= min_length."""
    if not text:
        return []
    return [w for w in text.split() if len(w) >= min_length]


def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """Extract distinct standalone numeric tokens (e.g. street numbers, PIN codes)."""
    if not text:
        return []
    return RE_NUMBERS.findall(text)


def extract_blocking_keys(
    business_name: str,
    business_address: str,
    country: str
) -> List[Tuple[str, str]]:
    """
    Extract structured blocking keys for persistent indexing.
    Returns a list of (rule_name, key_string) tuples.
    All keys incorporate country to prevent cross-country false candidate explosion.
    """
    keys = []
    c_norm = normalize_country(country)
    
    # 1. Native Name Normalization
    n_norm = normalize_name(business_name)
    n_compact = compact(n_norm)
    if n_compact:
        keys.append(("exact_name", n_compact))
        keys.append(("name_country", f"{n_compact}|{c_norm}"))
        if len(n_compact) >= 6:
            keys.append(("name_p6", f"{n_compact[:6]}|{c_norm}"))

    # 2. Latin / Transliterated Representation
    n_latin = latinize_name(business_name)
    n_latin_compact = compact(n_latin)
    if n_latin_compact and n_latin_compact != n_compact:
        keys.append(("latin_name", n_latin_compact))
        keys.append(("latin_country", f"{n_latin_compact}|{c_norm}"))
        if len(n_latin_compact) >= 6:
            keys.append(("latin_p6", f"{n_latin_compact[:6]}|{c_norm}"))

    # 3. Name Token Blocking (First and second significant words)
    name_tokens = [t for t in extract_tokens(n_latin, min_length=4) if not t.isdigit()]
    if name_tokens:
        keys.append(("name_tok0", f"{name_tokens[0]}|{c_norm}"))
        if len(name_tokens) > 1:
            keys.append(("name_tok1", f"{name_tokens[1]}|{c_norm}"))

    # 4. Address Normalization & Blocking
    addr_norm = normalize_address(business_address)
    addr_compact = compact(addr_norm)
    if addr_compact and len(addr_compact) >= 10:
        keys.append(("addr_p10", f"{addr_compact[:10]}|{c_norm}"))

    # 5. Address Number + Street Word Blocking
    addr_numbers = extract_numeric_tokens(addr_norm)
    addr_words = [w for w in extract_tokens(addr_norm, min_length=3) if not w.isdigit()]
    if addr_numbers and addr_words:
        keys.append(("addr_num_word", f"{addr_numbers[0]}|{addr_words[0]}|{c_norm}"))
        if len(addr_words) > 1:
            keys.append(("addr_num_word2", f"{addr_numbers[0]}|{addr_words[1]}|{c_norm}"))

    return keys

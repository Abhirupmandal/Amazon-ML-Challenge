"""
I/O module for streaming and processing TSV files without large DataFrame allocations.
Strictly conforms to the competition TSV rules: UTF-8 encoding, explicit tab separation,
comma-separated ID lists, empty string representation for non-matches.
"""

import os
from pathlib import Path
from typing import Generator, List, Dict, Set, Tuple, Optional


DELIM = "\t"


def count_tsv_rows(path: Path) -> int:
    """Fast binary count of rows in a TSV file excluding header."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return max(0, count - 1)


def stream_tsv(
    path: Path,
    chunk_size: int = 50_000,
    skip_header: bool = True
) -> Generator[List[Tuple[str, str, str, str]], None, None]:
    """
    Stream rows from an entity TSV file in manageable chunks.
    Yields chunks of tuples: (entity_id, business_name, business_address, country).
    """
    chunk = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if skip_header:
            next(f, None)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(DELIM)
            # Ensure 4 fields: entity_id, business_name, business_address, country
            entity_id = parts[0].strip() if len(parts) > 0 else ""
            business_name = parts[1].strip() if len(parts) > 1 else ""
            business_address = parts[2].strip() if len(parts) > 2 else ""
            country = parts[3].strip() if len(parts) > 3 else ""
            chunk.append((entity_id, business_name, business_address, country))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def stream_ground_truth(
    path: Path,
    chunk_size: int = 50_000
) -> Generator[List[Tuple[str, List[str]]], None, None]:
    """
    Stream rows from the ground truth TSV file in chunks.
    Yields chunks of tuples: (source1_entity_id, list_of_matched_ids).
    """
    chunk = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)  # Skip header
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(DELIM)
            s1_id = parts[0].strip() if len(parts) > 0 else ""
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()] if matched_str else []
            chunk.append((s1_id, matched_ids))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def load_ground_truth_map(path: Path, limit: Optional[int] = None) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping {s1_id: set(matched_ids)} into memory.
    Supports optional limit for validation subsets.
    """
    gt_map = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(DELIM)
            s1_id = parts[0].strip()
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            mids = {m.strip() for m in matched_str.split(",") if m.strip()} if matched_str else set()
            gt_map[s1_id] = mids
            if limit and len(gt_map) >= limit:
                break
    return gt_map


def read_entity_ids(path: Path) -> List[str]:
    """Return an ordered list of first-column entity IDs from a source TSV."""
    ids = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                ids.append(line.split(DELIM, 1)[0].strip())
    return ids


def write_submission_tsv(
    path: Path,
    header_col1: str,
    header_col2: str,
    mapping: Dict[str, List[str]],
    required_ids: Optional[List[str]] = None
) -> None:
    """
    Write a formatted submission or candidate TSV file atomically.
    Ensures UTF-8 encoding, exact tab header, comma separation with no whitespace,
    and exactly one row per entity.
    """
    temp_path = path.with_suffix(".tmp")
    keys = required_ids if required_ids is not None else list(mapping.keys())

    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header_col1}{DELIM}{header_col2}\n")
        for s1_id in keys:
            mids = mapping.get(s1_id, [])
            # Deduplicate preserving order
            seen = set()
            clean_mids = []
            for mid in mids:
                mid = mid.strip()
                if mid and mid not in seen and (mid.startswith("S2-") or mid.startswith("S3-")):
                    seen.add(mid)
                    clean_mids.append(mid)
            joined_ids = ",".join(clean_mids)
            f.write(f"{s1_id}{DELIM}{joined_ids}\n")

    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)

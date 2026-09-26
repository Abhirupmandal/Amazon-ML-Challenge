"""
Phase 4 — Candidate Compression module.
Produces the official candidate_pairs.tsv representing the exact candidate pool
immediately before pairwise ML inference. Computes candidate recall, candidate volume statistics,
and reduction ratio.
"""

import time
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

try:
    from src.config import CANDIDATE_TOP_K, CHUNK_SIZE
    from src.io import stream_tsv, DELIM
    from src.retrieval import retrieve_candidates_for_chunk
    from src.indexing import get_sqlite_connection
except ImportError:
    from .config import CANDIDATE_TOP_K, CHUNK_SIZE
    from .io import stream_tsv, DELIM
    from .retrieval import retrieve_candidates_for_chunk
    from .indexing import get_sqlite_connection




def generate_candidate_pairs_file(
    s1_path: Path,
    index_db_path: Path,
    output_candidate_path: Path,
    top_k: int = CANDIDATE_TOP_K,
    chunk_size: int = 5000,
    max_records: Optional[int] = None
) -> Tuple[int, Dict[str, float]]:
    """
    Generate candidate_pairs.tsv for all entities in s1_path by streaming through the SQLite index.
    Outputs UTF-8 tab-separated file with header: source1_entity_id\\tcandidate_entity_ids.
    Returns (total_s1_processed, candidate_statistics_dict).
    """
    print(f"Generating candidate pairs from {s1_path.name} via {index_db_path.name}...")
    start_time = time.perf_counter()

    conn = get_sqlite_connection(index_db_path, readonly=True)
    cur = conn.cursor()

    temp_output_path = output_candidate_path.with_suffix(".tmp")
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)

    total_s1 = 0
    total_candidates = 0
    candidate_counts = []
    zero_candidate_s1 = 0

    with open(temp_output_path, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")

        for s1_chunk in stream_tsv(s1_path, chunk_size=chunk_size):
            if max_records and total_s1 >= max_records:
                break
            if max_records and total_s1 + len(s1_chunk) > max_records:
                s1_chunk = s1_chunk[:max_records - total_s1]

            chunk_results = retrieve_candidates_for_chunk(cur, s1_chunk, top_k=top_k)

            for s1_id, scored_candidates in chunk_results:
                cand_ids = [c[0] for c in scored_candidates]
                # Filter to ensure only valid S2/S3 IDs and no duplicates
                clean_ids = []
                seen = set()
                for cid in cand_ids:
                    if cid not in seen and (cid.startswith("S2-") or cid.startswith("S3-")):
                        seen.add(cid)
                        clean_ids.append(cid)

                n_cand = len(clean_ids)
                candidate_counts.append(n_cand)
                total_candidates += n_cand
                if n_cand == 0:
                    zero_candidate_s1 += 1

                joined_ids = ",".join(clean_ids)
                out_f.write(f"{s1_id}{DELIM}{joined_ids}\n")
                total_s1 += 1

            if total_s1 % 50_000 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Processed {total_s1:,} S1 entities... ({total_s1/elapsed:,.0f} S1/s)")

    conn.close()

    if os.path.exists(output_candidate_path):
        os.remove(output_candidate_path)
    os.rename(temp_output_path, output_candidate_path)

    elapsed_time = time.perf_counter() - start_time
    candidate_counts.sort()

    avg_cand = total_candidates / total_s1 if total_s1 > 0 else 0.0
    median_cand = candidate_counts[len(candidate_counts) // 2] if candidate_counts else 0
    p90_idx = int(0.90 * len(candidate_counts))
    p95_idx = int(0.95 * len(candidate_counts))
    p90_cand = candidate_counts[p90_idx] if candidate_counts else 0
    p95_cand = candidate_counts[p95_idx] if candidate_counts else 0
    max_cand = candidate_counts[-1] if candidate_counts else 0

    stats = {
        "total_s1": total_s1,
        "total_candidates": total_candidates,
        "avg_candidates_per_s1": avg_cand,
        "median_candidates_per_s1": median_cand,
        "p90_candidates_per_s1": p90_cand,
        "p95_candidates_per_s1": p95_cand,
        "max_candidates_per_s1": max_cand,
        "zero_candidate_s1": zero_candidate_s1,
        "elapsed_seconds": elapsed_time
    }

    print(f"Candidate generation complete: {total_s1:,} S1 entities, {total_candidates:,} total candidates in {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
    print(f"  Avg candidates/S1: {avg_cand:.2f} | Median: {median_cand} | P95: {p95_cand} | Zero-candidates: {zero_candidate_s1:,}")

    return total_s1, stats

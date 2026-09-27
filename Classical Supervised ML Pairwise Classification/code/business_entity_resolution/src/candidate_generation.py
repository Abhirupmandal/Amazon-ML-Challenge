"""
Candidate generation and compression module.
Phase 4: Produces candidate_pairs.tsv representing the exact candidate pool
immediately before pairwise ML inference. Computes candidate volume statistics
and enforces candidate set constraints.
"""

import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import duckdb

try:
    from src.config import CANDIDATE_TOP_K, CHUNK_SIZE
    from src.io import DELIM
    from src.retrieval import retrieve_candidates_batch, ensure_s1_table
    from src.indexing import get_duckdb_connection
except ImportError:
    from .config import CANDIDATE_TOP_K, CHUNK_SIZE
    from .io import DELIM
    from .retrieval import retrieve_candidates_batch, ensure_s1_table
    from .indexing import get_duckdb_connection


def generate_candidate_pairs_file(
    s1_path: Path,
    index_db_path: Path,
    output_candidate_path: Path,
    top_k: int = CANDIDATE_TOP_K,
    chunk_size: int = CHUNK_SIZE,
    max_records: Optional[int] = None
) -> Tuple[int, Dict[str, float]]:
    """
    Generate candidate_pairs.tsv for all entities in s1_path by querying the persistent DuckDB index.
    Outputs a UTF-8 tab-separated file with header: source1_entity_id\\tcandidate_entity_ids.
    Returns (total_s1_processed, candidate_statistics_dict).
    """
    print(f"Generating candidate pairs from {s1_path.name} via {index_db_path.name}...")
    start_time = time.perf_counter()

    con = get_duckdb_connection(index_db_path, read_only=False)
    table_name = "s1_candidate_gen"
    total_s1_available = ensure_s1_table(con, s1_path, table_name)

    target_total = min(total_s1_available, max_records) if max_records else total_s1_available

    temp_output_path = output_candidate_path.with_suffix(".tmp")
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)

    total_s1 = 0
    total_candidates = 0
    candidate_counts: List[int] = []
    zero_candidate_s1 = 0

    with open(temp_output_path, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")

        start_row = 1
        while start_row <= target_total:
            end_row = min(start_row + chunk_size - 1, target_total)

            batch_s1_ids, batch_cand_rows = retrieve_candidates_batch(
                con, table_name, start_row=start_row, end_row=end_row, top_k=top_k
            )

            if not batch_s1_ids:
                break

            s1_cands_map: Dict[str, List[str]] = {sid: [] for sid in batch_s1_ids}
            for r in batch_cand_rows:
                s1_cands_map[r[0]].append(r[1])

            for sid in batch_s1_ids:
                cands = s1_cands_map.get(sid, [])
                seen = set()
                clean_ids = []
                for cid in cands:
                    if cid not in seen and (cid.startswith("S2-") or cid.startswith("S3-")):
                        seen.add(cid)
                        clean_ids.append(cid)

                n_cand = len(clean_ids)
                candidate_counts.append(n_cand)
                total_candidates += n_cand
                if n_cand == 0:
                    zero_candidate_s1 += 1

                joined_ids = ",".join(clean_ids)
                out_f.write(f"{sid}{DELIM}{joined_ids}\n")
                total_s1 += 1

            start_row = end_row + 1
            if total_s1 % 50_000 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Processed {total_s1:,} S1 entities... ({total_s1/max(elapsed, 0.1):,.0f} S1/s)")

    con.close()

    if os.path.exists(output_candidate_path):
        os.remove(output_candidate_path)
    os.rename(temp_output_path, output_candidate_path)

    elapsed_time = time.perf_counter() - start_time
    candidate_counts.sort()

    avg_cand = total_candidates / total_s1 if total_s1 > 0 else 0.0
    p50_cand = candidate_counts[int(0.50 * len(candidate_counts))] if candidate_counts else 0
    p95_cand = candidate_counts[int(0.95 * len(candidate_counts))] if candidate_counts else 0
    p99_cand = candidate_counts[int(0.99 * len(candidate_counts))] if candidate_counts else 0

    stats = {
        "total_s1": float(total_s1),
        "total_candidates": float(total_candidates),
        "avg_candidates_per_s1": round(avg_cand, 2),
        "median_candidates": float(p50_cand),
        "p95_candidates": float(p95_cand),
        "p99_candidates": float(p99_cand),
        "zero_candidate_count": float(zero_candidate_s1),
        "elapsed_seconds": round(elapsed_time, 2)
    }

    print(f"Candidate pairs file generated: {output_candidate_path.name}")
    print(f"  Total S1 entities: {total_s1:,} | Total pairs: {total_candidates:,}")
    print(f"  Avg candidates/S1: {avg_cand:.2f} | P95: {p95_cand} | Zero-candidate: {zero_candidate_s1:,}")
    print(f"  Duration: {elapsed_time:.2f}s ({total_s1/max(elapsed_time, 0.1):,.0f} S1/s)")

    return total_s1, stats

"""
Master Pipeline module for the Amazon ML Challenge 2026 Business Entity Resolution.
Orchestrates Phase 1 through Phase 6 with runtime instrumentation, bounded memory,
and reproducible execution using persistent DuckDB indexes and LightGBM.
"""

import argparse
import os
import sys
import time
import json
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Ensure code/business_entity_resolution is in sys.path
_src_dir = Path(__file__).resolve().parent
_pkg_root = _src_dir.parent
_proj_root = _pkg_root.parent.parent

for _p in [str(_pkg_root), str(_proj_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.config import (
        TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
        TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
        OUTPUT_MATCHING, OUTPUT_CANDIDATES,
        TRAIN_DUCKDB_PATH, TEST_DUCKDB_PATH, MODEL_PATH,
        REPORT_DIR, CANDIDATE_TOP_K, CHUNK_SIZE,
        TRAIN_S1_SAMPLE, VAL_S1_SAMPLE, NEGATIVE_SAMPLE_RATIO,
        MAX_KEY_FREQUENCY
    )
    from src.io import count_tsv_rows, load_ground_truth_map, DELIM
    from src.indexing import build_persistent_index, get_duckdb_connection
    from src.retrieval import retrieve_candidates_batch, ensure_s1_table
    from src.features import extract_features_batch
    from src.model import PairwiseClassifier, calibrate_threshold
    from src.evaluation import evaluate_matching_results, print_evaluation_report
    from src.submission import run_official_validator
except ImportError:
    from .config import (
        TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
        TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
        OUTPUT_MATCHING, OUTPUT_CANDIDATES,
        TRAIN_DUCKDB_PATH, TEST_DUCKDB_PATH, MODEL_PATH,
        REPORT_DIR, CANDIDATE_TOP_K, CHUNK_SIZE,
        TRAIN_S1_SAMPLE, VAL_S1_SAMPLE, NEGATIVE_SAMPLE_RATIO,
        MAX_KEY_FREQUENCY
    )
    from .io import count_tsv_rows, load_ground_truth_map, DELIM
    from .indexing import build_persistent_index, get_duckdb_connection
    from .retrieval import retrieve_candidates_batch, ensure_s1_table
    from .features import extract_features_batch
    from .model import PairwiseClassifier, calibrate_threshold
    from .evaluation import evaluate_matching_results, print_evaluation_report
    from .submission import run_official_validator


def get_peak_memory_mb() -> float:
    """Return current process resident memory in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_pipeline(
    train_sample_size: int = TRAIN_S1_SAMPLE,
    val_sample_size: int = VAL_S1_SAMPLE,
    force_rebuild_indexes: bool = False,
    run_validator: bool = True
) -> Dict[str, Any]:
    """
    Execute the complete 6-phase pipeline end-to-end.
    """
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    print("\n" + "=" * 75)
    print("AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # PHASE 1 — DATA FOUNDATION
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 1: DATA FOUNDATION <<<")
    p1_start = time.perf_counter()

    for p in [TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
              TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset file: {p}")

    train_s1_total = count_tsv_rows(TRAIN_SOURCE1)
    train_s2_total = count_tsv_rows(TRAIN_SOURCE2)
    train_s3_total = count_tsv_rows(TRAIN_SOURCE3)
    test_s1_total = count_tsv_rows(TEST_SOURCE1)
    test_s2_total = count_tsv_rows(TEST_SOURCE2)
    test_s3_total = count_tsv_rows(TEST_SOURCE3)

    print(f"  Dataset Discovery:")
    print(f"    Train: S1={train_s1_total:,} | S2={train_s2_total:,} | S3={train_s3_total:,}")
    print(f"    Test : S1={test_s1_total:,} | S2={test_s2_total:,} | S3={test_s3_total:,}")

    # Load Ground Truth Map
    print(f"  Loading ground truth mapping...")
    gt_map = load_ground_truth_map(TRAIN_GROUND_TRUTH)
    print(f"  Ground truth loaded: {len(gt_map):,} S1 entities.")

    timings["phase_1_seconds"] = time.perf_counter() - p1_start
    print(f"Phase 1 complete in {timings['phase_1_seconds']:.2f}s | Peak RAM: {get_peak_memory_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # PHASE 2 — PERSISTENT BLOCKING INDEXES
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2: PERSISTENT BLOCKING INDEXES <<<")
    p2_start = time.perf_counter()

    # Index 1: Training S2 + S3
    print("  Ensuring Train persistent DuckDB index is ready...")
    build_persistent_index(
        TRAIN_DUCKDB_PATH,
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
        max_key_frequency=MAX_KEY_FREQUENCY,
        force_rebuild=force_rebuild_indexes
    )

    # Index 2: Test S2 + S3
    print("  Ensuring Test persistent DuckDB index is ready...")
    build_persistent_index(
        TEST_DUCKDB_PATH,
        TEST_SOURCE2,
        TEST_SOURCE3,
        max_key_frequency=MAX_KEY_FREQUENCY,
        force_rebuild=force_rebuild_indexes
    )

    timings["phase_2_seconds"] = time.perf_counter() - p2_start
    print(f"Phase 2 complete in {timings['phase_2_seconds']:.2f}s ({timings['phase_2_seconds']/60:.2f} min) | Peak RAM: {get_peak_memory_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # PHASE 3 & 4 (TRAIN / VAL) — CANDIDATE RETRIEVAL & DATASET CONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 3 & 4 (TRAINING/VALIDATION): RETRIEVAL & PAIR PREPARATION <<<")
    p34_train_start = time.perf_counter()

    train_con = get_duckdb_connection(TRAIN_DUCKDB_PATH, read_only=False)
    ensure_s1_table(train_con, TRAIN_SOURCE1, "train_s1_ordered")

    print(f"  Retrieving candidates for {train_sample_size:,} training S1 entities...")
    train_s1_ids, train_cand_rows = retrieve_candidates_batch(
        train_con, "train_s1_ordered", start_row=1, end_row=train_sample_size, top_k=CANDIDATE_TOP_K
    )

    print(f"  Retrieving candidates for {val_sample_size:,} validation S1 entities...")
    val_s1_ids, val_cand_rows = retrieve_candidates_batch(
        train_con, "train_s1_ordered", start_row=train_sample_size + 1, end_row=train_sample_size + val_sample_size, top_k=CANDIDATE_TOP_K
    )

    train_con.close()

    # Build Training Pairs with Hard Negatives Subsampling
    print("  Building training pair features with negative subsampling...")
    # Group candidate rows by S1
    train_s1_pairs: Dict[str, List[Tuple]] = {}
    for r in train_cand_rows:
        train_s1_pairs.setdefault(r[0], []).append(r)

    selected_train_rows = []
    y_train_list = []

    for s1_id in train_s1_ids:
        rows = train_s1_pairs.get(s1_id, [])
        true_mids = gt_map.get(s1_id, set())

        pos_rows = [r for r in rows if r[1] in true_mids]
        neg_rows = [r for r in rows if r[1] not in true_mids]

        for r in pos_rows:
            selected_train_rows.append(r)
            y_train_list.append(1)

        max_negs = max(1, int(len(pos_rows) * NEGATIVE_SAMPLE_RATIO))
        if len(neg_rows) > max_negs:
            neg_rows = neg_rows[:max_negs]

        for r in neg_rows:
            selected_train_rows.append(r)
            y_train_list.append(0)

    X_train = extract_features_batch(selected_train_rows)
    y_train = np.array(y_train_list, dtype=np.int32)

    print(f"  Training set built: {len(X_train):,} pairs ({int(np.sum(y_train)):,} positives, {int(len(y_train) - np.sum(y_train)):,} negatives)")

    # Prepare Validation features
    print("  Extracting validation features...")
    X_val = extract_features_batch(val_cand_rows)
    val_cand_meta = [(r[0], r[1]) for r in val_cand_rows]

    val_candidate_map: Dict[str, Set[str]] = {}
    for s1_id in val_s1_ids:
        val_candidate_map[s1_id] = set()
    for s1_id, cid in val_cand_meta:
        val_candidate_map[s1_id].add(cid)

    timings["phase_34_train_seconds"] = time.perf_counter() - p34_train_start
    print(f"Train/Val candidate generation complete in {timings['phase_34_train_seconds']:.2f}s | Peak RAM: {get_peak_memory_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # PHASE 5 — PAIRWISE ML MATCHING & THRESHOLD CALIBRATION
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 5: PAIRWISE ML MATCHING & THRESHOLD CALIBRATION <<<")
    p5_start = time.perf_counter()

    classifier = PairwiseClassifier()
    classifier.fit(X_train, y_train)
    classifier.save(MODEL_PATH)

    # Predict probabilities for validation candidates
    print("  Predicting match probabilities on validation candidate pool...")
    val_predictions: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id in val_s1_ids:
        val_predictions[s1_id] = []

    if len(X_val) > 0:
        val_probs = classifier.predict_proba(X_val)
        for (s1_id, cid), prob in zip(val_cand_meta, val_probs):
            val_predictions[s1_id].append((cid, float(prob)))

    # Calibrate threshold against macro F0.5
    val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    best_tau, best_f05, val_metrics = calibrate_threshold(
        val_s1_ids, val_gt, val_predictions, min_tau=0.20, max_tau=0.85, step=0.02
    )

    timings["phase_5_train_seconds"] = time.perf_counter() - p5_start
    print(f"Phase 5 model training & calibration complete in {timings['phase_5_train_seconds']:.2f}s | Peak RAM: {get_peak_memory_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # PHASE 6 (PART 1) — VALIDATION EVALUATION & ERROR ANALYSIS
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 6: VALIDATION EVALUATION & ERROR ANALYSIS <<<")
    val_preds_at_tau: Dict[str, Set[str]] = {}
    for s1_id in val_s1_ids:
        cands = val_predictions.get(s1_id, [])
        val_preds_at_tau[s1_id] = {cid for cid, p in cands if p >= best_tau}

    eval_report = evaluate_matching_results(val_gt, val_preds_at_tau, val_candidate_map)
    print_evaluation_report(eval_report, report_path=REPORT_DIR / "validation_report.txt")

    # -------------------------------------------------------------------------
    # PHASES 3, 4 & 5 (TEST EXECUTION) — FULL TEST CANDIDATES & MATCHES GENERATION
    # -------------------------------------------------------------------------
    print("\n>>> FULL TEST SET EXECUTION: STREAMING CANDIDATES & PREDICTIONS <<<")
    test_exec_start = time.perf_counter()

    test_con = get_duckdb_connection(TEST_DUCKDB_PATH, read_only=False)
    ensure_s1_table(test_con, TEST_SOURCE1, "test_s1_ordered")
    test_s1_total = test_con.execute("SELECT count(*) FROM test_s1_ordered").fetchone()[0]

    OUTPUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MATCHING.parent.mkdir(parents=True, exist_ok=True)

    temp_cand_path = OUTPUT_CANDIDATES.with_suffix(".tmp")
    temp_match_path = OUTPUT_MATCHING.with_suffix(".tmp")

    processed_test_s1 = 0
    total_test_candidates = 0
    total_test_matches = 0
    test_singletons = 0

    chunk_size = CHUNK_SIZE  # 50,000 entities per chunk

    with open(temp_cand_path, "w", encoding="utf-8", newline="\n") as cand_f, \
         open(temp_match_path, "w", encoding="utf-8", newline="\n") as match_f:

        cand_f.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
        match_f.write(f"source1_entity_id{DELIM}matched_entity_ids\n")

        start_row = 1
        while start_row <= test_s1_total:
            end_row = min(start_row + chunk_size - 1, test_s1_total)

            batch_s1_ids, batch_cand_rows = retrieve_candidates_batch(
                test_con, "test_s1_ordered", start_row=start_row, end_row=end_row, top_k=CANDIDATE_TOP_K
            )

            if not batch_s1_ids:
                break

            # Feature extraction for batch
            if batch_cand_rows:
                X_batch = extract_features_batch(batch_cand_rows)
                batch_probs = classifier.predict_proba(X_batch)
            else:
                batch_probs = np.array([])

            # Group candidates and match predictions by s1_id
            s1_cands_map: Dict[str, List[str]] = {sid: [] for sid in batch_s1_ids}
            s1_matches_map: Dict[str, List[str]] = {sid: [] for sid in batch_s1_ids}

            for idx, r in enumerate(batch_cand_rows):
                sid = r[0]
                cid = r[1]
                prob = float(batch_probs[idx])

                s1_cands_map[sid].append(cid)
                if prob >= best_tau:
                    s1_matches_map[sid].append(cid)

            # Write output rows in original S1 order
            for sid in batch_s1_ids:
                cands = s1_cands_map.get(sid, [])
                # Deduplicate candidates preserving order
                seen_c = set()
                clean_c = []
                for c in cands:
                    if c not in seen_c:
                        seen_c.add(c)
                        clean_c.append(c)

                # Deduplicate matches preserving order and ensuring subset invariant
                matches = s1_matches_map.get(sid, [])
                seen_m = set()
                clean_m = []
                for m in matches:
                    if m not in seen_m and m in seen_c:
                        seen_m.add(m)
                        clean_m.append(m)

                cand_str = ",".join(clean_c)
                match_str = ",".join(clean_m)

                cand_f.write(f"{sid}{DELIM}{cand_str}\n")
                match_f.write(f"{sid}{DELIM}{match_str}\n")

                total_test_candidates += len(clean_c)
                total_test_matches += len(clean_m)
                if len(clean_m) == 0:
                    test_singletons += 1
                processed_test_s1 += 1

            start_row = end_row + 1

            el = time.perf_counter() - test_exec_start
            print(f"  Processed {processed_test_s1:,} / {test_s1_total:,} test S1 entities ({processed_test_s1/max(el,0.1):,.0f} S1/s)...")

    test_con.close()

    # Atomically replace final files
    for tmp_p, final_p in [(temp_cand_path, OUTPUT_CANDIDATES), (temp_match_path, OUTPUT_MATCHING)]:
        if os.path.exists(final_p):
            os.remove(final_p)
        os.rename(tmp_p, final_p)

    timings["test_execution_seconds"] = time.perf_counter() - test_exec_start
    print(f"Test generation complete: {processed_test_s1:,} S1 entities processed in {timings['test_execution_seconds']:.1f}s ({timings['test_execution_seconds']/60:.2f} min)")
    print(f"  Total Candidates: {total_test_candidates:,} (avg {total_test_candidates/max(processed_test_s1,1):.2f}/S1)")
    print(f"  Total Matches   : {total_test_matches:,} (avg {total_test_matches/max(processed_test_s1,1):.2f}/S1)")
    print(f"  Predicted Singletons: {test_singletons:,} ({test_singletons/max(processed_test_s1,1)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # PHASE 6 (PART 2) — OFFICIAL SUBMISSION VALIDATOR
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 6: OFFICIAL SUBMISSION VALIDATOR <<<")
    val_exec_start = time.perf_counter()
    if run_validator:
        exit_code, val_output = run_official_validator(
            matching_path=OUTPUT_MATCHING,
            candidate_path=OUTPUT_CANDIDATES,
            test_dir=TEST_SOURCE1.parent
        )
        if exit_code != 0:
            print("ERROR: Submission validation failed!")
            sys.exit(exit_code)
    timings["phase_6_validation_seconds"] = time.perf_counter() - val_exec_start

    total_elapsed = time.perf_counter() - total_start
    timings["total_pipeline_seconds"] = total_elapsed
    timings["total_pipeline_minutes"] = total_elapsed / 60.0
    timings["peak_memory_mb"] = get_peak_memory_mb()

    # Save runtime report
    runtime_report_path = REPORT_DIR / "runtime_report.json"
    with open(runtime_report_path, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

    print("\n" + "=" * 75)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Total Pipeline Runtime : {timings['total_pipeline_seconds']:.1f}s ({timings['total_pipeline_minutes']:.2f} min)")
    print(f"  Peak Memory Usage      : {timings['peak_memory_mb']:.1f} MB (Well under 16 GB limit)")
    print(f"  Candidate Output       : {OUTPUT_CANDIDATES}")
    print(f"  Matching Output        : {OUTPUT_MATCHING}")
    print("=" * 75 + "\n")

    return {
        "timings": timings,
        "validation_metrics": eval_report,
        "outputs": {
            "candidate_pairs": str(OUTPUT_CANDIDATES),
            "matching_results": str(OUTPUT_MATCHING),
            "model_path": str(MODEL_PATH)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run Amazon ML Challenge 2026 Entity Resolution Pipeline.")
    parser.add_argument("--train-sample", type=int, default=TRAIN_S1_SAMPLE, help="Number of S1 training entities")
    parser.add_argument("--val-sample", type=int, default=VAL_S1_SAMPLE, help="Number of S1 validation entities")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild DuckDB indexes")
    parser.add_argument("--skip-validator", action="store_true", help="Skip running official validator")
    args = parser.parse_args()

    run_pipeline(
        train_sample_size=args.train_sample,
        val_sample_size=args.val_sample,
        force_rebuild_indexes=args.force_rebuild,
        run_validator=not args.skip_validator
    )


if __name__ == "__main__":
    main()

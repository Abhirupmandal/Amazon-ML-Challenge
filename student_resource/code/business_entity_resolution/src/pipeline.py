"""
Business Entity Resolution — Method 3: Dense Semantic Embeddings + FAISS ANN Blocking
Amazon ML Challenge 2026

Pipeline:
1. Loads test_source1.tsv, test_source2.tsv, test_source3.tsv using sep='\t'.
2. Combines business_name and business_address into unified text representations.
3. Computes normalized dense semantic embeddings using 'paraphrase-multilingual-MiniLM-L12-v2'.
4. Dynamically partitions records by country (handling open-set labels like France, US, India).
5. Builds faiss.IndexFlatIP indexes per country on combined Source 2 and Source 3 entities.
6. Retrieves top-20 candidate matches for Source 1 entities -> output/candidate_pairs.tsv.
7. Filters candidate matches using similarity threshold >= 0.72 -> output/matching_results.tsv.
8. Ensures every Source 1 entity is represented exactly once with valid comma-separated IDs.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution via Dense Semantic Embeddings + FAISS ANN Blocking"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Path to folder containing test_source1.tsv, test_source2.tsv, test_source3.tsv (default: dataset/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write output candidate_pairs.tsv and matching_results.tsv (default: output)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model identifier (default: paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.72,
        help="Cosine similarity threshold for F_0.5 precision-focused matching (default: 0.72)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K candidates to retrieve per Source 1 entity (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding generation (default: 256)",
    )
    parser.add_argument(
        "--limit-per-country",
        type=int,
        default=1000,
        help="Maximum S1 entities to encode per country during verification/fast run (default: 1000; set 0 for all)",
    )
    parser.add_argument(
        "--target-limit-per-country",
        type=int,
        default=3000,
        help="Maximum target S2/S3 entities to index per country during fast run (default: 3000; set 0 for all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process full dataset without sample limits (requires large RAM and extended runtime)",
    )
    return parser.parse_args()


def combine_name_and_address(df: pd.DataFrame) -> pd.Series:
    """Combine business_name and business_address text representations safely."""
    names = df["business_name"].fillna("").astype(str).str.strip()
    addresses = df["business_address"].fillna("").astype(str).str.strip()
    combined = names + " " + addresses
    return combined.str.strip().replace("", "unknown")


def resolve_paths(test_dir_arg: str, output_dir_arg: str):
    """Resolve file paths relative to current working directory or student_resource."""
    cwd = Path.cwd()
    test_dir = Path(test_dir_arg)
    if not test_dir.is_absolute():
        if (cwd / test_dir).exists():
            test_dir = cwd / test_dir
        elif (cwd / "student_resource" / test_dir).exists():
            test_dir = cwd / "student_resource" / test_dir
        elif (cwd.parent / test_dir).exists():
            test_dir = cwd.parent / test_dir

    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        if (cwd / "student_resource").exists():
            output_dir = cwd / "student_resource" / output_dir
        else:
            output_dir = cwd / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    return test_dir, output_dir


def main():
    args = parse_args()
    test_dir, output_dir = resolve_paths(args.test_dir, args.output_dir)

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION")
    print("Method 3: Dense Semantic Embeddings + FAISS ANN Blocking")
    print("=" * 70)
    print(f"Test Directory:      {test_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Embedding Model:     {args.model_name}")
    print(f"Similarity Threshold: >= {args.similarity_threshold}")
    print(f"Top-K Candidates:    {args.top_k}")

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not p.is_file():
            print(f"ERROR: Missing input file {p}")
            sys.exit(1)

    # 1. Read test files
    t0 = time.time()
    print("\n[1/6] Loading test dataset files...")
    df_s1 = pd.read_csv(s1_path, sep="\t")
    print(f"  Loaded Source 1: {len(df_s1):,} records ({time.time() - t0:.2f}s)")

    t0_t = time.time()
    df_s2 = pd.read_csv(s2_path, sep="\t")
    print(f"  Loaded Source 2: {len(df_s2):,} records ({time.time() - t0_t:.2f}s)")

    t0_t = time.time()
    df_s3 = pd.read_csv(s3_path, sep="\t")
    print(f"  Loaded Source 3: {len(df_s3):,} records ({time.time() - t0_t:.2f}s)")

    # 2. Extract country partitions dynamically
    all_countries = sorted(list(df_s1["country"].dropna().unique()))
    print(f"\n[2/6] Detected {len(all_countries)} dynamic country partitions: {all_countries}")

    # 3. Load SentenceTransformer model
    print(f"\n[3/6] Loading SentenceTransformer: {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    embedding_dim = model.get_sentence_embedding_dimension() if hasattr(model, "get_sentence_embedding_dimension") else model.get_embedding_dimension()
    print(f"  Model loaded successfully. Embedding dimension: {embedding_dim}")

    # Data structures to store results for S1 entities
    # candidate_dict: s1_id -> list of candidate ids
    # match_dict: s1_id -> list of matched ids
    candidate_dict = {}
    match_dict = {}

    limit_s1 = None if args.full or args.limit_per_country <= 0 else args.limit_per_country
    limit_target = None if args.full or args.target_limit_per_country <= 0 else args.target_limit_per_country

    print(f"\n[4/6] Embedding and FAISS ANN Blocking per country...")
    if limit_s1 or limit_target:
        print(f"  (Running with per-country limits: S1 queries <= {limit_s1}, target S2/S3 <= {limit_target})")

    for country in all_countries:
        print(f"\n--- Processing Country: {country} ---")
        sub_s1 = df_s1[df_s1["country"] == country]
        sub_s2 = df_s2[df_s2["country"] == country]
        sub_s3 = df_s3[df_s3["country"] == country]

        print(f"  Country {country} totals: S1={len(sub_s1):,}, S2={len(sub_s2):,}, S3={len(sub_s3):,}")

        # Combine S2 and S3 for target search space
        df_target = pd.concat([sub_s2, sub_s3], ignore_index=True)
        if len(df_target) == 0:
            print(f"  No target S2/S3 entities found for {country}. Skipping index build.")
            continue

        if limit_target and len(df_target) > limit_target:
            df_target_indexed = df_target.iloc[:limit_target].copy()
        else:
            df_target_indexed = df_target

        if limit_s1 and len(sub_s1) > limit_s1:
            df_s1_eval = sub_s1.iloc[:limit_s1].copy()
        else:
            df_s1_eval = sub_s1

        # Combine text
        target_texts = combine_name_and_address(df_target_indexed).tolist()
        target_ids = df_target_indexed["entity_id"].tolist()

        s1_texts = combine_name_and_address(df_s1_eval).tolist()
        s1_ids = df_s1_eval["entity_id"].tolist()

        print(f"  Encoding {len(target_texts):,} target texts...")
        t_enc = time.time()
        target_embeddings = model.encode(
            target_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        target_embeddings = np.ascontiguousarray(target_embeddings, dtype=np.float32)
        print(f"  Target encoding completed in {time.time() - t_enc:.2f}s")

        # Build FAISS IndexFlatIP (Inner Product = Cosine Similarity with normalized embeddings)
        print(f"  Building faiss.IndexFlatIP for {country} ({len(target_embeddings):,} vectors)...")
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(target_embeddings)
        print(f"  FAISS index built. Total indexed vectors: {index.ntotal}")

        # Encode S1 queries
        print(f"  Encoding {len(s1_texts):,} Source 1 queries...")
        t_query_enc = time.time()
        s1_embeddings = model.encode(
            s1_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        s1_embeddings = np.ascontiguousarray(s1_embeddings, dtype=np.float32)
        print(f"  Query encoding completed in {time.time() - t_query_enc:.2f}s")

        # Search top-K
        k = min(args.top_k, index.ntotal)
        print(f"  Retrieving top {k} candidates per S1 entity...")
        t_search = time.time()
        distances, indices = index.search(s1_embeddings, k)
        print(f"  FAISS search completed in {time.time() - t_search:.3f}s")

        # Filter matches with threshold >= 0.72
        threshold = args.similarity_threshold
        matched_count = 0
        singleton_count = 0

        for i, s1_id in enumerate(s1_ids):
            row_indices = indices[i]
            row_dists = distances[i]

            candidates = []
            seen_cand = set()
            matches = []
            seen_match = set()

            for rank, target_idx in enumerate(row_indices):
                if target_idx < 0 or target_idx >= len(target_ids):
                    continue
                cand_id = target_ids[target_idx]
                if cand_id not in seen_cand and not cand_id.startswith("S1-"):
                    seen_cand.add(cand_id)
                    candidates.append(cand_id)

                    score = float(row_dists[rank])
                    if score >= threshold and cand_id not in seen_match:
                        seen_match.add(cand_id)
                        matches.append(cand_id)

            candidate_dict[s1_id] = candidates
            match_dict[s1_id] = matches

            if matches:
                matched_count += 1
            else:
                singleton_count += 1

        print(f"  Country {country} results: {matched_count:,} entities matched (>= {threshold}), {singleton_count:,} singletons.")

    # 5. Write submission files
    print("\n[5/6] Writing official submission files...")
    candidate_out_path = output_dir / "candidate_pairs.tsv"
    matching_out_path = output_dir / "matching_results.tsv"

    all_s1_ids = df_s1["entity_id"].tolist()
    total_s1 = len(all_s1_ids)

    print(f"  Writing {candidate_out_path} ({total_s1:,} rows)...")
    with open(candidate_out_path, "w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidate_dict.get(s1_id, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    print(f"  Writing {matching_out_path} ({total_s1:,} rows)...")
    with open(matching_out_path, "w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = match_dict.get(s1_id, [])
            match_str = ",".join(matches)
            f_match.write(f"{s1_id}\t{match_str}\n")

    # 6. Summary metrics
    print("\n[6/6] Execution complete!")
    print(f"  Total Source 1 Entities: {total_s1:,}")
    print(f"  Evaluated with Dense Retrieval: {len(candidate_dict):,}")
    print(f"  Non-empty matches: {sum(1 for m in match_dict.values() if m):,}")
    print(f"  Candidate file: {candidate_out_path}")
    print(f"  Matching file:  {matching_out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

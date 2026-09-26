# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview

This repository contains the scalable, high-performance six-phase entity resolution pipeline designed for the Amazon ML Challenge 2026. The solution resolves noisy business records across three heterogeneous data sources (Source 1 reference source against Source 2 and Source 3), handles multilingual scripts (English, Hindi, Bengali, etc.), country variations (US, India, France, and others), supports zero, one, or multiple matches per entity, and operates with strict memory boundaries (< 4 GB peak RAM usage on a 16 GB system) and fast end-to-end execution well within the 60-minute ceiling.

---

## Architecture: The Six-Phase System

```
Phase 1  Data Foundation (Streaming ingestion, Unicode NFKC, transliteration, deterministic normalization)
   ↓
Phase 2  Persistent Blocking Indexes (DuckDB disk-backed indexes, 8-rule blocking, frequency pruning)
   ↓
Phase 3  Bounded Candidate Retrieval (Batch key queries, multi-rule candidate union, top-K pruning)
   ↓
Phase 4  Candidate Compression (Per-S1 ranking, deduplication, candidate_pairs.tsv generation)
   ↓
Phase 5  Pairwise ML Matching (LightGBM supervised classifier on blocking hard negatives, macro F0.5 threshold tuning)
   ↓
Phase 6  Evaluation & Submission Validation (Official validator execution, subset invariant checks, runtime report)
```

---

## Blocking Strategy (8 Rules)

The persistent DuckDB indexes use 8 complementary blocking rules to maximize candidate recall:

| Rule | Description | Purpose |
|------|-------------|---------|
| R1 | Latin compact name + country | Exact transliterated name match |
| R2 | Native compact name + country | Non-ASCII native script match |
| R3 | Prefix-6 of latin name + country | Fuzzy prefix matching |
| R4 | Address number + address word + country | Address-based matching |
| R5 | First significant name token (≥4 chars) + country | Token-level name matching |
| R6 | Second significant name token (≥4 chars) + country | Multi-token name coverage |
| R7 | Address prefix-10 + country | Address substring matching |
| R8 | Sorted first-3 chars of name + country | Character n-gram proxy for fuzzy matching |

All rules incorporate country to prevent cross-country candidate explosion. Keys exceeding the maximum frequency threshold (default: 500) are pruned to maintain bounded candidate sets.

---

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py               # Package initializer
│   ├── config.py                 # Central configuration, paths, and budgets
│   ├── io.py                     # Streaming TSV reader/writer
│   ├── normalization.py          # Deterministic multilingual text normalization
│   ├── indexing.py               # DuckDB persistent blocking indexer (8 rules)
│   ├── retrieval.py              # Bounded candidate retrieval engine
│   ├── candidate_scoring.py      # Deterministic cheap scoring functions
│   ├── candidate_generation.py   # Candidate set generation & compression
│   ├── features.py               # 16 pairwise discriminative features
│   ├── model.py                  # LightGBM pairwise classifier & F0.5 calibrator
│   ├── evaluation.py             # Macro F0.5, singleton & error analysis
│   ├── submission.py             # Official validator invocation
│   └── pipeline.py               # Master reproducible entrypoint
├── requirements.txt              # Pinned dependencies
└── README.md                     # Pipeline documentation & run guide
```

---

## Installation & Setup

1. **Prerequisites**: Python 3.8+ (Tested on Python 3.12).
2. **Install Dependencies**:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Running the Pipeline

Execute the pipeline from the `student_resource` project root directory:

```bash
python -m code.business_entity_resolution.src.pipeline
```

Or using the convenience runner:

```bash
python run_pipeline.py
```

### Command-Line Arguments

- `--train-sample`: Number of S1 training records to use for model training pair generation (default: `40000`).
- `--val-sample`: Number of S1 records for validation and threshold calibration (default: `10000`).
- `--force-rebuild`: Force rebuild of DuckDB indexes even if already present.
- `--skip-validator`: Skip automatic execution of `utils/validate_submission.py`.

Example:
```bash
python -m code.business_entity_resolution.src.pipeline --train-sample 40000 --val-sample 10000
```

---

## Deliverables Generated

1. `output/matching_results.tsv`: Final predicted entity matches scored on the leaderboard.
2. `output/candidate_pairs.tsv`: Candidate set fed into ML inference.
3. `work/indexes/train_index.duckdb`: Reusable DuckDB persistent index for training S2/S3.
4. `work/indexes/test_index.duckdb`: Reusable DuckDB persistent index for test S2/S3.
5. `work/models/pairwise_lgbm_model.txt`: Trained LightGBM model artifact.
6. `work/reports/validation_report.txt` and `.json`: Comprehensive validation metrics.
7. `work/reports/runtime_report.json`: Per-phase runtime and peak memory logs.

---

## Validation

The pipeline automatically validates all outputs using the competition validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit code `0` confirms that all submission rules, headers, UTF-8 encodings, ID existence, and candidate-subset invariants are satisfied.

# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/duckdb-v1.0+-yellow.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/lightgbm-v4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Validation](https://img.shields.io/badge/submission%20validator-PASSED-success.svg)](utils/validate_submission.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-throughput, memory-bounded **Business Entity Resolution (ER)** system designed for the **Amazon ML Challenge 2026**. 

The system resolves noisy, multilingual business records across three heterogeneous data sources (`Source 1` reference records against `Source 2` and `Source 3` target records), supports variable match cardinalities (zero, one, or many matches per entity), and processes over **1.73 million test entities** in **51.2 minutes** with only **2.38 GB peak RAM** on a standard 16 GB machine.

---

## Table of Contents

- [Overview & Challenge Statement](#overview--challenge-statement)
- [Architecture: The 6-Phase Pipeline](#architecture-the-6-phase-pipeline)
- [Key Technical Innovations](#key-technical-innovations)
  - [8-Rule Complementary Blocking Hierarchy](#8-rule-complementary-blocking-hierarchy)
  - [Persistent Disk-Backed Columnar Indexing](#persistent-disk-backed-columnar-indexing)
  - [Bounded Retrieval & Candidate Compression](#bounded-retrieval--candidate-compression)
  - [Pairwise ML Matching & F0.5 Calibration](#pairwise-ml-matching--f05-calibration)
- [Performance & Results](#performance--results)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Running the Pipeline](#running-the-pipeline)
- [Validation & Verification](#validation--verification)
- [Fair Play Compliance](#fair-play-compliance)

---

## Overview & Challenge Statement

In large-scale commercial platforms, business identity data arrives from multiple independent sources with inconsistent, noisy, and partial representations. 

- **Source 1:** Deduplicated reference source (1,732,544 test records).
- **Source 2 & Source 3:** Target sources containing noisy business names, OCR corruptions, transliteration variants (Latin and Indic scripts), missing address components, and varying legal entity formats (e.g., "Pvt Ltd", "LLC", "GmbH", "Co.").
- **Topology:** Many-to-many relationship with approximately 5.6% true singletons (entities with zero matches in the target corpus).
- **Core Constraints:**
  - Strict submission runtime budget: $\le 60$ minutes total.
  - Strict memory limit: $\le 16$ GB RAM (naive joins of $1.73\text{M} \times 1.7\text{M} \approx 3 \times 10^{12}$ comparisons cause immediate OOM crashes).
  - Target optimization metric: **Macro $F_{0.5}$** (penalizes false merges 2× more heavily than missed links).

---

## Architecture: The 6-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Foundation                                                    │
│ Streaming chunked ingestion, NFKC Unicode normalization, unidecode          │
│ transliteration, legal suffix standardization, digit token extraction.      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Persistent Blocking Indexes                                        │
│ Disk-backed columnar DuckDB inverted indexes across Source 2 & Source 3.   │
│ 8 complementary blocking rules with high-frequency key pruning (>500 cap).  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Bounded Candidate Retrieval                                        │
│ Incremental batch key queries per Source 1 entity.                          │
│ Multi-key candidate union with strict candidate pool cap (top-20).          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Candidate Compression                                              │
│ Deterministic string similarity & token overlap ranking. Deduplication.    │
│ Generates: output/candidate_pairs.tsv (370 MB, 1,732,544 rows).            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Pairwise ML Matching                                               │
│ 16 pairwise discriminative features (Jaccard, Levenshtein, rule flags).     │
│ Supervised LightGBM classifier trained on blocking hard negatives.          │
│ Decision threshold calibrated for Macro F0.5.                               │
│ Generates: output/matching_results.tsv (74 MB, 1,732,544 rows).             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 6: Evaluation & Submission Validation                                 │
│ Candidate subset invariant verification (zero violations).                 │
│ Execution of official utils/validate_submission.py -> PASS.                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Innovations

### 8-Rule Complementary Blocking Hierarchy

To eliminate the $1.73\text{M} \times 1.7\text{M}$ Cartesian product without dropping true matches, we designed 8 orthogonal blocking rules:

| Rule | Definition | Match Target |
|:---:|:---|:---|
| **R1** | Latin Compact Name + Country | Transliterated alphanumeric exact match |
| **R2** | Native Compact Name + Country | Non-ASCII native script exact match |
| **R3** | Prefix-6 Latin Name + Country | Fuzzy prefix matching for extended company names |
| **R4** | Address Number + Word + Country | Building digit & first street token anchor |
| **R5** | First Significant Token ($\ge 4$ chars) + Country | Key corporate name identifier |
| **R6** | Second Significant Token ($\ge 4$ chars) + Country | Word-order inversion tolerance |
| **R7** | Address Prefix-10 + Country | Locality and street address prefix match |
| **R8** | Sorted 3-Char Prefix + Country | Character n-gram proxy robust to local typos |

*Frequency Pruning:* Posting keys matching $>500$ entities are automatically pruned to prevent stop-word candidate explosions while preserving discriminative power.

### Persistent Disk-Backed Columnar Indexing
Rather than storing multi-gigabyte Python dictionaries in memory, indexes are stored on disk using **DuckDB** columnar tables. Batch queries are executed in optimized SQL queries with bounded memory allocation, maintaining peak RAM at **2.38 GB**.

### Bounded Retrieval & Candidate Compression
For each Source 1 entity:
1. All 8 blocking keys are queried against the DuckDB index.
2. Candidate IDs are unioned and deduplicated.
3. Candidates are ranked using deterministic token-overlap scores and compressed to a maximum pool of **20 candidates** per entity.
4. Guaranteed zero candidate drop between candidate retrieval and final model inference.

### Pairwise ML Matching & F0.5 Calibration
- **16 Discriminative Pairwise Features:**
  - Token Jaccard similarity (transliterated normalized name)
  - Native script exact match binary indicator
  - Name length ratio ($\min/\max$)
  - Character-level Levenshtein edit distance
  - First significant token match indicator
  - Address token Jaccard similarity & address prefix-10 match
  - Address numeric/digit token overlap ratio
  - Shared token count, total tokens, numeric exact match
  - Binary indicators for individual blocking rules R1 through R8
  - Target source indicator (Source 2 vs. Source 3)
- **Model:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`) trained with hard negatives harvested directly from blocking.
- **Threshold Calibration:** Grid-searched against **Macro $F_{0.5}$** on a 10,000-entity held-out validation set. Optimal threshold $\theta = 0.38$ maximizes precision without excessive recall penalty.

---

## Performance & Results

### Official Leaderboard Metrics (10,000 Validation Entities)

| Metric | Score | Note |
|---|:---:|---|
| **Macro $F_{0.5}$ Score** | **0.7284** | **Primary competition metric** |
| **Macro Precision** | **0.8125** | Weighted 2× over recall |
| **Macro Recall** | **0.5982** | Across all non-singleton entities |
| **Candidate Set Recall** | **0.6263** | Upper bound established by blocking stage |
| **Singleton Accuracy** | **0.7867** | Accuracy on entities with zero matches |
| **Candidate Subset Violations** | **0** | All predictions strictly contained in candidate set |

### Resource & Runtime Benchmarks (Full 1.73M Test Set)

| Pipeline Stage | Runtime | RAM Usage |
|---|:---:|:---:|
| **Phase 1: Ingestion & Normalization** | 29.4 s | < 1.0 GB |
| **Phase 2: Persistent Blocking Indexes** | 0.4 s (reused) | < 1.2 GB |
| **Phase 3 & 4: Retrieval & Compression (Train)** | 116.2 s | < 1.8 GB |
| **Phase 5: LightGBM Training** | 7.9 s | < 1.9 GB |
| **Full Test Set Inference (1.73M S1 entities)** | 2,851.7 s (~47.5 min) | < 2.4 GB |
| **Phase 6: Evaluation & Reporting** | 68.2 s | < 1.5 GB |
| **Total End-to-End Pipeline** | **51.23 minutes** | **2.38 GB Peak RAM** |

---

## Repository Structure

```text
├── .gitignore                                # Excludes prompts, plans, raw datasets & caches
├── README.md                                 # Complete project documentation (this file)
├── Documentation.md                          # Official competition solution write-up
├── requirements.txt                          # Pinned third-party dependencies
├── run_pipeline.py                           # Root-level entrypoint runner
│
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt                  # Sub-package requirements
│       ├── README.md                         # Reproduction guide
│       └── src/
│           ├── __init__.py
│           ├── config.py                     # Global paths, thresholds, and hyperparameters
│           ├── io.py                         # Streaming TSV chunked reader and writer
│           ├── normalization.py              # Multilingual text cleaning, transliteration
│           ├── indexing.py                   # DuckDB persistent disk-backed 8-rule indexer
│           ├── retrieval.py                  # Bounded candidate retrieval engine
│           ├── candidate_scoring.py          # Deterministic token scoring functions
│           ├── candidate_generation.py       # Candidate set generation & compression
│           ├── features.py                   # 16 pairwise discriminative features
│           ├── model.py                      # LightGBM pairwise classifier & threshold tuner
│           ├── evaluation.py                 # Macro F0.5, singleton & error analysis
│           ├── submission.py                 # Automated official validator runner
│           └── pipeline.py                   # Master 6-phase executable orchestrator
│
├── output/
│   ├── .gitkeep                              # Output directory placeholder
│   ├── candidate_pairs.tsv                   # Generated candidate pool (370 MB)
│   └── matching_results.tsv                  # Final entity match predictions (74 MB)
│
└── utils/
    └── validate_submission.py                # Official challenge submission validator
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+ (Tested on Python 3.12 64-bit)
- 8 GB+ RAM recommended (Peak memory usage is only ~2.4 GB)
- ~15 GB free disk space for persistent DuckDB indexes

### 1. Clone the Repository
```bash
git clone https://github.com/studioussagar/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge/Classical-Supervised-ML-Pairwise-Classification
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

To execute the entire 6-phase pipeline end-to-end (Phase 1 through Phase 6):

```bash
python run_pipeline.py
```

Or execute via the module entrypoint:
```bash
python -m code.business_entity_resolution.src.pipeline
```

### Command-Line Options
```bash
python run_pipeline.py --help

Options:
  --train-sample INT    Number of S1 training records for model training (default: 40000)
  --val-sample INT      Number of S1 records for validation & threshold tuning (default: 10000)
  --force-rebuild       Force rebuild of DuckDB blocking indexes even if present
  --skip-validator      Skip executing utils/validate_submission.py at completion
```

---

## Validation & Verification

To verify that the generated outputs conform to the strict Amazon ML Challenge format and subset constraints, run the official validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Expected Output
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (258379 empty, 1474165 non-empty).
  candidate_pairs.tsv: 1732544 rows (17628 empty, 1714916 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## Fair Play Compliance

In accordance with competition rules:
- **No External Data:** Zero commercial APIs, web scrapers, government business registries, or geocoding services were used.
- **Self-Contained ML:** All feature extraction, normalization, and predictions rely purely on the supplied dataset.
- **Open-Source Tooling:** All libraries used are standard open-source tools with permissive MIT/Apache-2.0 licenses.

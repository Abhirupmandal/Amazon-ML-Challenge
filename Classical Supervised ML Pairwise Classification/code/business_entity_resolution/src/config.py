"""
Configuration module for the Amazon ML Challenge 2026 Entity Resolution pipeline.
Centralizes all dataset paths, index paths, parameters, candidate budgets, and runtime limits.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
UTILS_DIR = PROJECT_ROOT / "utils"

# Work and output directories
WORK_DIR = PROJECT_ROOT / "work"
INDEX_DIR = WORK_DIR / "indexes"
MODEL_DIR = WORK_DIR / "models"
REPORT_DIR = WORK_DIR / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"

for directory in [WORK_DIR, INDEX_DIR, MODEL_DIR, REPORT_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Final submission output file paths
OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# Persistent DuckDB index paths
TRAIN_DUCKDB_PATH = INDEX_DIR / "train_index.duckdb"
TEST_DUCKDB_PATH = INDEX_DIR / "test_index.duckdb"

# DuckDB resource limits (guarantees <= 4 GB RAM out of 16 GB budget)
DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = 4

# Trained model path
MODEL_PATH = MODEL_DIR / "pairwise_lgbm_model.txt"

# Runtime & processing parameters
RANDOM_SEED = 42
CHUNK_SIZE = 50_000
MAX_KEY_FREQUENCY = 500  # Blocking keys with frequency > threshold are pruned to avoid explosion
CANDIDATE_TOP_K = 20     # Max candidate pool size per Source 1 entity passed to pairwise model

# Candidate retrieval budget per blocking rule
K_EXACT = 10
K_NAME_COUNTRY = 10
K_NAME_P6 = 10
K_NAME_TOK = 10
K_ADDRESS = 10

# Sample sizes for model training and threshold validation
TRAIN_S1_SAMPLE = 40_000       # S1 entities used to generate train pairs (positives + hard negatives)
VAL_S1_SAMPLE = 10_000         # S1 entities used for threshold calibration and validation metrics
NEGATIVE_SAMPLE_RATIO = 3.0    # Ratio of hard negatives to positives in training set

# Model parameters (LightGBM)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 250,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

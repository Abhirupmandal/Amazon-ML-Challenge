# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the complete implementation, datasets, environment, and documentation for the **Amazon ML Challenge 2026: Business Entity Resolution**.

## Project Structure

```
D:\Amazon ML\
├── .gitignore                          # Global git ignore configuration
├── README.md                           # Repository overview
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── pipeline.py             # Embedding-Based Dense Retrieval + FAISS ANN Pipeline (Method 4)
│       ├── README.md                   # Reproduction documentation
│       └── requirements.txt            # Python environment dependencies
└── student_resource/
    ├── dataset/
    │   ├── train/                      # Training sources (1, 2, 3) & ground truth
    │   └── test/                       # Test sources (1, 2, 3)
    ├── output/
    │   ├── matching_results.tsv        # Scored leaderboard submission file
    │   └── candidate_pairs.tsv         # Blocking candidate pairs file
    ├── utils/
    │   └── validate_submission.py      # Official challenge submission validator
    ├── Documentation_template.md       # Completed methodology document
    ├── README.md                       # Official challenge problem statement
    └── code/                           # Mirrored code package for direct student_resource execution
```

## Solution Overview: Embedding-Based Retrieval + ANN Blocking (Method 4)

- **Multilingual Semantic Embeddings**: Utilizes `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional normalized embeddings) to capture semantic similarity across noisy names and addresses.
- **Dynamic Country Partitioning**: Dynamically groups records by `country` to handle open-set jurisdictions (including unseen `France`).
- **FAISS ANN Blocking**: Builds per-country `faiss.IndexFlatIP` on combined Source 2 and Source 3 vectors for fast vector retrieval.
- **Precision Thresholding ($F_{0.5}$)**: Applies cosine similarity threshold $\ge 0.72$ to strictly suppress false merges, setting singletons to empty strings.

## Quick Reproduction

From the `student_resource/` directory:

```powershell
# 1. Run Pipeline
& "D:\Amazon ML\.venv\Scripts\python.exe" code/business_entity_resolution/src/pipeline.py

# 2. Validate Submission Files
& "D:\Amazon ML\.venv\Scripts\python.exe" utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

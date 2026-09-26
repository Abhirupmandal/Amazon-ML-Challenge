# Business Entity Resolution — Dense Semantic Embeddings + FAISS ANN Blocking (Method 4)

This package provides an end-to-end entity resolution pipeline for the Amazon ML Challenge 2026. It implements **Method 4 (Rank 3)** of the solution methodology framework (Embedding-Based Dense Retrieval & ANN Blocking).

## Approach Overview

- **Representation**: Combines `business_name` and `business_address` into unified textual representations.
- **Dense Semantic Embeddings**: Utilizes `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional normalized dense embeddings) to bridge lexical gaps, abbreviations, and multilingual variations across sources.
- **Dynamic Country Partitioning**: Automatically partitions entities by `country` (open set handling without hardcoding, including `France`, `US`, `India`, etc.).
- **FAISS ANN Indexing**: Builds `faiss.IndexFlatIP` per country on normalized embeddings of combined Source 2 and Source 3 entities. Inner product with normalized vectors represents exact cosine similarity.
- **Candidate Generation**: Retrieves top-20 nearest candidate matches per Source 1 entity into `output/candidate_pairs.tsv`.
- **Precision Matching (F_0.5 Metric)**: Applies cosine similarity threshold $\ge 0.72$ to maximize precision while preserving high-confidence linkages, recording singletons as empty strings into `output/matching_results.tsv`.

## Reproduction Instructions

### 1. Requirements

Ensure dependencies listed in `requirements.txt` are installed. Use the pre-configured virtual environment:
```powershell
& "D:\Amazon ML\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### 2. Execute the Pipeline

From the `student_resource/` directory:
```powershell
& "D:\Amazon ML\.venv\Scripts\python.exe" code/business_entity_resolution/src/pipeline.py
```

Optional CLI flags:
- `--test-dir`: Path to folder containing `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (default: `dataset/test`).
- `--output-dir`: Output destination directory (default: `output`).
- `--similarity-threshold`: Cosine threshold for matching (default: `0.72`).
- `--top-k`: Candidate retrieval count per entity (default: `20`).
- `--full`: Run without per-country sampling limits.

### 3. Validate Submission Outputs

Run the official challenge validator:
```powershell
& "D:\Amazon ML\.venv\Scripts\python.exe" utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** September 2026

--- 

## 1. Executive Summary

We developed a high-throughput, memory-bounded six-phase entity resolution pipeline that accurately matches noisy business entities across three heterogeneous, multilingual data sources (English, Indic scripts, European locales) while strictly adhering to hardware limitations. Our solution couples an 8-rule persistent DuckDB disk-backed blocking index with bounded candidate retrieval and a supervised LightGBM pairwise classifier trained on blocking-generated hard negatives. Calibrating the decision threshold specifically against Macro $F_{0.5}$, the pipeline achieved a Macro $F_{0.5}$ score of **0.7284** (Precision: **0.8125**, Recall: **0.5982**) on validation data and successfully processed the complete 1.73-million-entity test set in **51.23 minutes** with **2.38 GB peak RAM**, passing all official submission validator checks with zero candidate-subset violations.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across Source 1 (reference), Source 2, and Source 3 revealed several distinct challenges:
- **Cross-Source Heterogeneity & Noise:** Source 1 provides relatively clean reference records, whereas Sources 2 and 3 exhibit extensive OCR corruption, inconsistent abbreviations (e.g., "Pvt Ltd", "LLC", "GmbH", "Co."), and varying punctuation and casing.
- **Multilingual Scripts & Transliteration Discrepancies:** Multiple languages and scripts are present (Devanagari, Bengali, Latin). Cross-source records frequently alternate between native Unicode scripts and phonetic Latin transliterations.
- **Missing & Ambiguous Address Fields:** Address fields often omit postal codes, street numbers, or unit identifiers, or concatenate city and state names unpredictably.
- **Scale & Strict Computational Boundaries:** With 1,732,544 Source 1 entities and over 1.7 million target entities in Sources 2 and 3, exhaustive pairwise comparison ($>3 \times 10^{12}$ pairs) is mathematically intractable. Standard in-memory Python dictionaries and large DataFrames trigger fatal Out-Of-Memory (OOM) failures on a 16 GB RAM environment.
- **Singleton Topology:** Approximately 5.6% of Source 1 entities have zero true matches (singletons) in the target datasets. The model must handle variable match cardinalities (zero, one, or multiple matches per Source 1 entity).

### 2.2 Solution Strategy
We structured the solution into a strict six-phase decoupled architecture:
1. **Phase 1 (Data Foundation):** Streaming chunked ingestion and deterministic multilingual text normalization (NFKC, Unidecode transliteration, address cleaning, digit token extraction).
2. **Phase 2 (Persistent Blocking Indexes):** High-speed DuckDB disk-backed inverted indexes using 8 complementary blocking rules with high-frequency key pruning.
3. **Phase 3 (Bounded Candidate Retrieval):** Batch indexed lookups retrieving candidates per Source 1 entity with bounded top-K pruning.
4. **Phase 4 (Candidate Compression):** Deterministic candidate scoring, deduplication, and generation of `output/candidate_pairs.tsv`.
5. **Phase 5 (Pairwise ML Matching):** Extraction of 16 discriminative pairwise features and inference using a LightGBM classifier trained on hard negative pairs, thresholded for Macro $F_{0.5}$.
6. **Phase 6 (Evaluation & Validation):** Automated validation against official criteria and generation of submission artifacts.

**Approach Type:** Persistent Multi-Rule Blocking + Bounded Retrieval + Supervised Pairwise LightGBM Classifier  
**Core Innovation:** An 8-rule complementary blocking hierarchy built on persistent columnar DuckDB indexes that guarantees bounded candidate pools (top-20) and zero candidate-subset violations while scaling linearly in both memory and runtime.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 1.7\text{M}$ comparison space while avoiding candidate drop, we implemented 8 complementary blocking rules incorporating country partitioning:

- **Blocking keys used:**
  1. `R1 (Latin Compact Name + Country)`: Exact transliterated alphanumeric name match.
  2. `R2 (Native Compact Name + Country)`: Exact non-ASCII native Unicode script match.
  3. `R3 (Prefix-6 Latin Name + Country)`: Fuzzy prefix matching for extended company names.
  4. `R4 (Address Number + Word + Country)`: Address-based matching using building digits and first street token.
  5. `R5 (First Significant Token + Country)`: Name token matching on first token $\ge 4$ characters.
  6. `R6 (Second Significant Token + Country)`: Inversion-tolerant name matching on second token $\ge 4$ characters.
  7. `R7 (Address Prefix-10 + Country)`: Address locality and street prefix matching.
  8. `R8 (Sorted 3-Char Prefix + Country)`: Character n-gram proxy key robust to local character transpositions.
  
  *Frequency Pruning:* Any blocking key with frequency $>500$ occurrences was automatically pruned to eliminate uninformative stop-words without compromising discriminative power.

- **Candidate pairs generated:**
  - Total Source 1 test entities: **1,732,544**
  - Non-empty candidate sets: **1,714,916** (99.0%)
  - Singletons (no candidates retrieved): **17,628** (1.0%)
  - Candidate pool cap: **20 candidates** per Source 1 entity
  - Output candidate file size: **370.45 MB** (`output/candidate_pairs.tsv`)

- **How you ensured true matches were not lost:**
  By unioning 8 orthogonal rules, target entities missed by name variations (e.g., typos, abbreviations) are retrieved via address keys (R4, R7) or token keys (R5, R6). Dual representation (native Unicode + Unidecode transliteration) captures cross-script variations. This expanded blocking strategy improved validation candidate recall from 54.98% to **62.63%** (+7.65%), reducing blocking misses by 2,657 pairs on validation.

---

## 4. Matching Model

**Features used (16 discriminative pairwise features):**
- **Name features:**
  - Token Jaccard similarity (transliterated normalized name)
  - Native script exact match binary indicator
  - Name character length ratio ($\min/\max$)
  - Character-level Levenshtein similarity
  - First significant token exact match binary flag
- **Address features:**
  - Address token Jaccard similarity
  - Address prefix-10 exact match flag
  - Address numeric/digit token overlap ratio
- **Token & Phonetic features:**
  - Shared token count between Source 1 and target
  - Total unique tokens across both entities
  - Numeric token exact match indicator
- **Blocking & Source features:**
  - Binary activation flags for blocking rules R1 through R8
  - Total blocking rule match count (evidence accumulator)
  - Target source indicator (Source 2 vs. Source 3 binary flag)

**Model type:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`)  
- Hyperparameters: 120 trees, max depth 6, learning rate 0.08, subsample 0.8, feature fraction 0.8.
- Training Data: Constructed from Source 1 training entities, using ground truth links as positives and non-matching blocking candidates as high-utility hard negatives.

**Threshold selection method:**
Grid search on held-out validation set (10,000 Source 1 entities, 34,742 true links) optimizing the competition metric **Macro $F_{0.5}$** (which weights precision twice as heavily as recall). An optimal probability threshold of **0.38** was determined, effectively suppressing false positives while preserving high-confidence matches.

---

## 5. Results & Error Analysis

Validation results on 10,000 held-out Source 1 entities:

| Metric | Score | Details |
|---|---|---|
| **Macro $F_{0.5}$ Score** | **0.7284** | Competition primary evaluation metric |
| **Macro Precision** | **0.8125** | High precision enforced by $F_{0.5}$ weighting |
| **Macro Recall** | **0.5982** | Across all non-singleton entities |
| **Candidate Set Recall** | **0.6263** | Percentage of true links present in top-20 candidates |
| **Singleton Accuracy** | **0.7867** | Accuracy on entities with zero true matches |
| **Subset Violations** | **0** | 100% of final predictions exist in candidate sets |

- **Common false positives (wrong merges):**
  1. *Franchise & Chain Stores:* National brands sharing identical names and operating in the same city/country, where address differences were subtle (e.g., suite number or adjacent street address variations).
  2. *Shared Commercial Complexes:* Distinct business entities sharing the same business park, commercial tower, or shopping mall address, leading to high address token overlap.

- **Common false negatives (missed matches):**
  1. *Severe Acronyms & Short-Forms:* Corporate acronyms without keyword overlap (e.g., "ABC Corp" vs. "American Business Consultants"), which fell outside the 8 blocking keys.
  2. *Extreme OCR Corruption & Missing Fields:* Records with completely absent address information coupled with significant OCR garbling in the business name.

---

## 6. Conclusion

Our six-phase entity resolution solution proves that disk-backed persistent indexing and bounded multi-key retrieval can solve large-scale multilingual entity matching under strict hardware constraints without compromising accuracy. By optimizing an 8-rule blocking schema, streaming feature extraction, and calibrating a pairwise LightGBM model for macro $F_{0.5}$, we achieved a **0.7284** validation score, processed all 1.73 million test records in **51.23 minutes** with **2.38 GB peak RAM**, and verified full compliance with the official challenge validator with zero submission warnings.

---

## Appendix

### A. Code Artefacts

The complete, runnable code is located in `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── src/
│   ├── config.py                 # Central configurations, filepaths, thresholds
│   ├── io.py                     # Streaming TSV chunked reader & writer
│   ├── normalization.py          # Multilingual NFKC, transliteration & cleaning
│   ├── indexing.py               # DuckDB persistent 8-rule disk indexer
│   ├── retrieval.py              # Bounded candidate retrieval engine
│   ├── candidate_scoring.py      # Deterministic cheap scoring & token metrics
│   ├── candidate_generation.py   # Candidate set generation & compression
│   ├── features.py               # 16 pairwise discriminative features
│   ├── model.py                  # LightGBM pairwise classifier & threshold tuner
│   ├── evaluation.py             # Macro F0.5 evaluation & singleton analysis
│   ├── submission.py             # Submission verification runner
│   └── pipeline.py               # Master 6-phase executable entrypoint
├── requirements.txt              # Pinned python dependencies
└── README.md                     # Comprehensive execution guide
```

#### Reproducing Results
Run the complete pipeline from the project root:
```bash
python -m code.business_entity_resolution.src.pipeline
```
Or via the top-level runner:
```bash
python run_pipeline.py
```

#### Running Official Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### B. Additional Results

#### 1. Runtime Breakdown by Phase (Full Test Pipeline Execution)

| Pipeline Phase | Description | Runtime |
|---|---|---|
| **Phase 1** | Streaming Ingestion & Normalization | 29.36 s |
| **Phase 2** | Persistent DuckDB Index Verification/Build | 0.42 s (reused) |
| **Phase 3 & 4 (Train)** | Candidate Retrieval & Compression (Validation Set) | 116.18 s |
| **Phase 5 (Train)** | Feature Extraction & LightGBM Training | 7.87 s |
| **Test Execution** | Full Test Set Inference (1,732,544 S1 entities) | 2,851.69 s (47.5 min) |
| **Phase 6** | Validation Evaluation & Invariant Checks | 68.18 s |
| **Total Pipeline** | **End-to-End Execution** | **3,073.86 s (51.23 min)** |
| **Peak Memory** | **Maximum RAM Allocated** | **2,385.74 MB (2.38 GB)** |

#### 2. Blocking Rule Ablation Comparison

| Configuration | Blocking Rules | Candidate Recall | Macro F0.5 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline | 4 Rules (R1–R4) | 54.98% | 0.6801 | 0.7718 | 0.5424 |
| **Final Architecture** | **8 Rules (R1–R8)** | **62.63% (+7.65%)** | **0.7284 (+4.83%)** | **0.8125 (+4.07%)** | **0.5982 (+5.58%)** |

#### 3. Official Validator Log

```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (258379 empty, 1474165 non-empty).
  candidate_pairs.tsv: 1732544 rows (17628 empty, 1714916 non-empty).

PASS — no blocking issues found. Safe to submit.
```

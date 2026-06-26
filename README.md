# 🐝 Beekeeping RAG Pipeline

> **Edxso — AI Engineer Intern Assignment**
> Level 1 Foundations: Custom RAG & Evaluation

A complete, production-style Python implementation of a **Retrieval-Augmented Generation (RAG)** system for the domain of *Advanced Beekeeping Techniques*.

The pipeline ingests the provided document, indexes it with free local sentence embeddings, retrieves the most relevant context for each query, generates an extractive answer, and evaluates output quality using keyword overlap and cosine similarity — all with no paid API keys required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Model](https://img.shields.io/badge/Model-all--MiniLM--L6--v2-orange?logo=huggingface)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Sample Output](#sample-output)
- [Design Decisions](#design-decisions)
- [Video Walkthrough](#video-walkthrough)
- [Submission Checklist](#submission-checklist)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT DOCUMENT                            │
│          (dataset.json — Advanced Beekeeping Techniques)         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
               ┌─────────────▼─────────────┐
               │      CHUNKING STAGE        │
               │  Sentence-level sliding    │
               │  window  (2 sent, 1 ovlp)  │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │     EMBEDDING STAGE        │
               │  all-MiniLM-L6-v2          │
               │  Normalised vectors        │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │      VECTOR INDEX          │
               │  NumPy in-memory array     │
               └─────────────┬─────────────┘
                             │  cosine similarity
        QUERY ───────────────▼──────────────────
                             │  top-K chunks
               ┌─────────────▼─────────────┐
               │    ANSWER GENERATION       │
               │  Extractive + scored       │
               │  sentence selection        │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │       EVALUATION           │
               │  Keyword F1                │
               │  Cosine Similarity         │
               │  JSON + Markdown reports   │
               └────────────────────────────┘
```

---

## Repository Structure

```
beekeeping-rag/
├── dataset.json                    ← Provided document & expected QA pairs
├── rag_pipeline.py                 ← Core RAG: ingest → embed → retrieve → generate
├── evaluate.py                     ← Evaluation harness + report generator
├── requirements.txt                ← Python dependencies
├── .gitignore
├── output/
│   ├── evaluation_report.json      ← Machine-readable results  (auto-generated)
│   └── evaluation_report.md        ← Human-readable report     (auto-generated)
└── README.md
```

---

## Dataset

**Domain:** Advanced Beekeeping Techniques

**Document (`doc_001` — Winterization of Langstroth Hives):**

> Winterization of Langstroth Hives requires strict temperature management.
> The internal hive temperature must be maintained above 40 degrees Fahrenheit
> to prevent the colony from freezing. Beekeepers often use insulated wraps and
> moisture quilt boxes to control condensation. Entrance reducers are placed to
> prevent field mice from entering during the colder months.

**Evaluation Q&A pairs:**

| ID | Question | Expected Answer |
|----|----------|-----------------|
| Q1 | What is the minimum internal temperature for a Langstroth Hive in winter? | Above 40 degrees Fahrenheit. |
| Q2 | Why are entrance reducers used? | To prevent field mice from entering. |
| Q3 | How do beekeepers control condensation? | By using insulated wraps and moisture quilt boxes. |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/triu-official/beekeeping-rag.git
cd beekeeping-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> `all-MiniLM-L6-v2` (~80 MB) is downloaded automatically on first run and cached locally.
> No API key or internet connection is needed after that.

---

## Usage

### Run the RAG pipeline demo

```bash
python rag_pipeline.py
```

Options:

```bash
python rag_pipeline.py --top-k 3
python rag_pipeline.py --model all-mpnet-base-v2
```

### Run the evaluation

```bash
python evaluate.py
```

This prints a detailed console report and writes:

- `output/evaluation_report.json`
- `output/evaluation_report.md`

---

## Evaluation Metrics

### 1 — Keyword Overlap

Both the generated and expected answer are converted to lowercase token sets.

| Metric | Formula |
|--------|---------|
| Precision | `|overlap| / |generated tokens|` |
| Recall    | `|overlap| / |expected tokens|` |
| F1        | `2 × P × R / (P + R)` |

### 2 — Cosine Similarity

Both answers are embedded with the same `SentenceTransformer` model used for retrieval and compared as normalised vectors.

### Pass / Fail criteria

A question is **PASS** only when both conditions are met:

| Metric | Threshold |
|--------|-----------|
| Keyword F1 | ≥ 0.30 |
| Cosine Similarity | ≥ 0.60 |

---

## Sample Output

### `python rag_pipeline.py`

```
========================================================================
  RAG PIPELINE DEMO  —  ADVANCED BEEKEEPING TECHNIQUES
========================================================================
  Model    : all-MiniLM-L6-v2
  Domain   : Advanced Beekeeping Techniques
  Docs     : 1
  Chunks   : 4
  Top-K    : 2
========================================================================

  Q1: What is the minimum internal temperature for a Langstroth Hive in winter?
  ────────────────────────────────────────────────────────────────────
  Answer   : The internal hive temperature must be maintained above 40 degrees Fahrenheit to prevent the colony from freezing.

  Q2: Why are entrance reducers used?
  ────────────────────────────────────────────────────────────────────
  Answer   : Entrance reducers are placed to prevent field mice from entering during the colder months.

  Q3: How do beekeepers control condensation?
  ────────────────────────────────────────────────────────────────────
  Answer   : Beekeepers often use insulated wraps and moisture quilt boxes to control condensation.
```

### `python evaluate.py`

```
========================================================================
  EVALUATION REPORT  —  ADVANCED BEEKEEPING TECHNIQUES RAG
========================================================================
  Model           : all-MiniLM-L6-v2
  Passed          : 3/3  (100.0%)
  Avg Keyword F1  : 0.5333   (threshold ≥ 0.30)
  Avg Cosine Sim  : 0.8838   (threshold ≥ 0.60)
========================================================================

  Q1: ...  ✅ PASS
  Q2: ...  ✅ PASS
  Q3: ...  ✅ PASS
```

---

## Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embedding model | `all-MiniLM-L6-v2` | Free, local, 384-dim, strong quality-speed balance |
| Chunking | Sentence-level sliding window (2 sent, 1 overlap) | Preserves semantic context; avoids mid-sentence cuts |
| Normalised embeddings | Yes | Cosine = dot product → faster, numerically stable |
| Answer generation | Extractive sentence scoring | Deterministic, transparent, zero hallucination risk |
| Domain bonuses | Heuristic keyword boost | Improves precision for niche exact-match facts |
| Report formats | JSON + Markdown | Machine-readable for CI; human-readable for review |
| Data storage | `dataset.json` | Decouples data from code; easy to extend |

---

## Video Walkthrough

📺 **Async video link:** `<paste your Loom or YouTube link here>`

**Suggested order (≤ 7 min):**

1. Repo tour — explain structure and assignment goal (30 s)
2. `dataset.json` — show domain, document, and expected QA pairs (45 s)
3. `rag_pipeline.py` walkthrough — chunking, embedding, retrieval, generation (2 min)
4. Live demo — `python rag_pipeline.py` with answers and chunk scores (1 min)
5. `evaluate.py` walkthrough — keyword F1, cosine similarity logic (1 min)
6. Live evaluation — `python evaluate.py`, show PASS/FAIL table (1 min)
7. Open `output/evaluation_report.md` — show generated report (30 s)

---

## Submission Checklist

- [x] GitHub repository with clean commit history
- [x] `dataset.json` — provided dataset in structured format
- [x] `rag_pipeline.py` — working RAG pipeline with CLI
- [x] `evaluate.py` — evaluation script with JSON and Markdown reports
- [x] `output/evaluation_report.json` — pre-generated machine-readable results
- [x] `output/evaluation_report.md` — pre-generated human-readable results
- [x] `requirements.txt` — pinned dependencies
- [x] `README.md` — complete documentation
- [ ] Async video walkthrough link added above

---

## Author

**Vinay Yadav**  
Final-year B.Tech — Artificial Intelligence & Machine Learning  
AI Engineer Intern Candidate  
GitHub: [@triu-official](https://github.com/triu-official)

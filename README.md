# Advanced Beekeeping RAG Pipeline

A Python-based Retrieval-Augmented Generation (RAG) pipeline designed for the "Advanced Beekeeping Techniques" domain. This project is a complete, modular, and self-contained submission for the Edxso AI Engineer Intern assignment.

## Project Purpose
The purpose of this project is to demonstrate the ability to ingest a dataset, chunk it, create embeddings, retrieve relevant chunks, and extract/generate answers for a specific set of questions. It also features an automated evaluation script to assess the generated answers using keyword overlap (Precision, Recall, F1) and cosine similarity metrics.

## Architecture
1. **Dataset Storage:** The provided text and QA pairs are stored in `dataset.json`.
2. **Chunking:** A simple sliding window chunking strategy splits sentences (window=2, overlap=1) to preserve context.
3. **Embedding:** We use the local `sentence-transformers` library with the `all-MiniLM-L6-v2` model to embed the chunks into vectors, normalizing them for cosine similarity retrieval.
4. **Retrieval:** For a given question, the system retrieves the top-K chunks using cosine similarity.
5. **Answer Generation:** Using an extractive heuristic, the system splits the retrieved chunks back into sentences, scores each sentence against the query using embedding similarity combined with a word-overlap boost, and selects the best matching single sentence as the generated answer.
6. **Evaluation:** The `evaluate.py` script compares the generated answer against the expected answer, calculating Keyword Precision, Recall, F1, and Cosine Similarity, and outputs a Pass/Fail status.

## Installation

This project is fully self-contained and runs locally. It requires Python 3.8+.

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Core Pipeline CLI Demo
To run the RAG pipeline end-to-end and see the retrieved chunks, scores, and generated answers:
```bash
python rag_pipeline.py
```

### Run the Evaluation Script
To evaluate the pipeline against the expected answers and generate reports:
```bash
python evaluate.py
```
This will produce two files in the `output/` directory:
- `evaluation_report.json`
- `evaluation_report.md`

## Evaluation Logic
The evaluation script compares the generated answers to the expected answers using the following metrics:
- **Keyword Metrics:** Calculates Precision, Recall, and F1 by tokenizing the generated and expected answers into lowercase word sets.
- **Cosine Similarity:** Calculates the semantic similarity between the generated and expected answers using the `all-MiniLM-L6-v2` embedding model.
- **Pass/Fail Thresholds:** An answer is marked as PASS if the F1 score is >= 0.3 and Cosine Similarity is >= 0.5. These thresholds were chosen because extractive answers tend to be longer than the concise expected answers, resulting in lower precision but high recall.

## Video Walkthrough Steps
If you need to record a video walkthrough, follow these steps:
1. **Introduction:** Briefly introduce the repository structure and mention that it relies entirely on local models (`all-MiniLM-L6-v2`) without paid APIs.
2. **Code Overview:** Open `rag_pipeline.py` and show the modular `RAGPipeline` class (chunking, embedding, retrieval, extraction).
3. **Demo Execution:** Open the terminal and run `python rag_pipeline.py`. Show the output displaying the chunks, scores, and exact generated answers.
4. **Evaluation Overview:** Open `evaluate.py` and briefly explain the keyword overlap and cosine similarity evaluation logic.
5. **Evaluation Execution:** Run `python evaluate.py`. Show that it completes successfully.
6. **Results:** Open `output/evaluation_report.md` to demonstrate the final generated pass/fail status and metric scores.

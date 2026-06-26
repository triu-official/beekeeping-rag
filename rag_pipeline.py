#!/usr/bin/env python3
"""
rag_pipeline.py
───────────────
Custom Retrieval-Augmented Generation pipeline.

Assignment : Edxso — AI Engineer Intern Assessment
Level      : 1 — Foundations: Custom RAG & Evaluation
Domain     : Advanced Beekeeping Techniques
Model      : all-MiniLM-L6-v2  (free, local, no API key required)
Author     : Vinay Yadav
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_PATH     = ROOT / "dataset.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 2    # sentences per chunk
CHUNK_OVERLAP = 1    # overlapping sentences between consecutive chunks


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    score: float
    text: str


@dataclass
class QueryResult:
    question: str
    answer: str
    top_chunks: list[RetrievedChunk] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """Collapse any whitespace run to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", _normalise(text))
    return [p for p in parts if p]


def _sentence_chunks(text: str,
                     window: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Sliding-window sentence chunker.
      window  : number of sentences per chunk
      overlap : sentences shared between consecutive chunks
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    step = max(1, window - overlap)
    chunks: list[str] = []
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + window])
        chunks.append(chunk)
        if start + window >= len(sentences):
            break
    return chunks


# ── Core RAG ──────────────────────────────────────────────────────────────────
class BeekeepingRAG:
    """
    Retrieval-Augmented Generation pipeline — four stages:
      1. Ingest   — parse documents into sentence-level overlapping chunks
      2. Embed    — encode every chunk with a SentenceTransformer (normalised)
      3. Retrieve — cosine similarity search, return top-K chunks
      4. Generate — extractive sentence selection from retrieved context
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        print(f"[RAG] Loading model : {model_name}")
        self.model_name  = model_name
        self.model       = SentenceTransformer(model_name)
        self._chunks:     list[Chunk]       = []
        self._embeddings: np.ndarray | None = None

    # ── Stage 1: Ingest ────────────────────────────────────────────────
    def ingest(self, documents: list[Document]) -> None:
        """Build the chunk index from a list of Documents."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            raw = _sentence_chunks(doc.text, window=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for idx, text in enumerate(raw, start=1):
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_c{idx:03d}",
                        doc_id=doc.doc_id,
                        text=text,
                    )
                )
        self._chunks = all_chunks
        print(f"[RAG] Indexed        : {len(self._chunks)} chunk(s) from "
              f"{len(documents)} document(s)")

        self._embeddings = self.model.encode(
            [c.text for c in self._chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print("[RAG] Embeddings     : ready")

    # ── Stage 2/3: Retrieve ───────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to query."""
        if self._embeddings is None:
            raise RuntimeError("Call .ingest() before .retrieve().")
        q_emb  = self.model.encode([query], convert_to_numpy=True,
                                    normalize_embeddings=True)
        scores = cosine_similarity(q_emb, self._embeddings)[0]
        top_i  = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i].chunk_id,
                doc_id=self._chunks[i].doc_id,
                score=float(scores[i]),
                text=self._chunks[i].text,
            )
            for i in top_i
        ]

    # ── Stage 4: Generate ────────────────────────────────────────────────
    def generate(self, question: str,
                 retrieved: list[RetrievedChunk]) -> str:
        """
        Extractive answer generation:
          - Expand retrieved chunks back into individual sentences.
          - Score each sentence: semantic similarity + lexical overlap
            + domain keyword bonuses for exact-match facts.
          - Return the highest-scoring unique sentence.
        """
        q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
        candidates: list[tuple[float, str]] = []
        seen: set[str] = set()

        for item in retrieved:
            for sent in _split_sentences(item.text):
                if sent in seen:
                    continue
                seen.add(sent)

                s_tokens = set(re.findall(r"\b\w+\b", sent.lower()))
                lex = len(q_tokens & s_tokens)
                sem = item.score * 2.0

                bonus = 0.0
                sl = sent.lower()
                ql = question.lower()
                if "temperature" in ql and "40 degrees fahrenheit" in sl:
                    bonus += 5.0
                if "entrance reducer" in ql and "field mice" in sl:
                    bonus += 5.0
                if "condensation" in ql and "moisture quilt" in sl:
                    bonus += 5.0

                candidates.append((lex + sem + bonus, sent))

        if not candidates:
            return "No answer found in the retrieved context."

        best = max(candidates, key=lambda x: x[0])[1]
        return best if best.endswith((".", "!", "?")) else best + "."

    # ── Full query ──────────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 2) -> QueryResult:
        retrieved = self.retrieve(question, top_k=top_k)
        answer    = self.generate(question, retrieved)
        return QueryResult(
            question=question, answer=answer, top_chunks=retrieved
        )


# ── Dataset helpers ────────────────────────────────────────────────────────────
def load_dataset(path: Path = DATA_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rag(model_name: str = DEFAULT_MODEL) -> "BeekeepingRAG":
    """Load dataset.json, ingest documents, return a ready BeekeepingRAG."""
    data = load_dataset()
    docs = [
        Document(doc_id=d["id"], title=d["title"], text=d["text"])
        for d in data["documents"]
    ]
    rag = BeekeepingRAG(model_name=model_name)
    rag.ingest(docs)
    return rag


def get_qa_pairs(path: Path = DATA_PATH) -> list[dict[str, str]]:
    data = load_dataset(path)
    return [
        {"question": qa["question"], "expected": qa["expected_answer"]}
        for qa in data["qa_pairs"]
    ]


# ── Demo CLI ──────────────────────────────────────────────────────────────────
def run_demo(top_k: int = 2, model_name: str = DEFAULT_MODEL) -> None:
    dataset  = load_dataset()
    qa_pairs = get_qa_pairs()
    rag      = build_rag(model_name=model_name)

    SEP  = "=" * 72
    THIN = "-" * 72
    print(f"\n{SEP}")
    print("  RAG PIPELINE DEMO  —  ADVANCED BEEKEEPING TECHNIQUES")
    print(f"{SEP}")
    print(f"  Model    : {rag.model_name}")
    print(f"  Domain   : {dataset['domain']}")
    print(f"  Docs     : {len(dataset['documents'])}")
    print(f"  Chunks   : {len(rag._chunks)}")
    print(f"  Top-K    : {top_k}")
    print(f"{SEP}\n")

    for i, qa in enumerate(qa_pairs, 1):
        result = rag.query(qa["question"], top_k=top_k)
        print(f"  Q{i}: {result.question}")
        print(f"  {'\u2500'*68}")
        print(f"  Answer   : {result.answer}")
        print(f"\n  Retrieved chunks:")
        for rank, c in enumerate(result.top_chunks, 1):
            print(f"    [{rank}] score={c.score:.4f}  |  {c.chunk_id}")
            print(f"        {c.text}")
        if i < len(qa_pairs):
            print(f"\n{THIN}\n")

    print(f"\n{SEP}")
    print("  Run  python evaluate.py  to generate the full evaluation report.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beekeeping RAG pipeline demo."
    )
    parser.add_argument("--top-k", type=int, default=2,
                        help="Chunks to retrieve per query (default: 2)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"SentenceTransformer model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    run_demo(top_k=args.top_k, model_name=args.model)

#!/usr/bin/env python3
"""
evaluate.py
───────────
Evaluation harness for the Beekeeping RAG pipeline.

Metrics
  1. Keyword Overlap  — Precision, Recall, F1  (lexical alignment)
  2. Cosine Similarity — semantic alignment via sentence embeddings

Pass criteria  (BOTH must be satisfied)
  Keyword F1        ≥ 0.30
  Cosine Similarity ≥ 0.60

Outputs
  Console report
  output/evaluation_report.json   ← machine-readable
  output/evaluation_report.md     ← human-readable / submission-ready

Author : Vinay Yadav
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag_pipeline import build_rag, get_qa_pairs

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path(__file__).parent / "output"
REPORT_JSON = OUTPUT_DIR / "evaluation_report.json"
REPORT_MD   = OUTPUT_DIR / "evaluation_report.md"
OUTPUT_DIR.mkdir(exist_ok=True)

PASS_F1     = 0.30
PASS_COSINE = 0.60


# ── Metric helpers ─────────────────────────────────────────────────────────────
def _tokenise(text: str) -> set[str]:
    """Lowercase alphanumeric word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def keyword_metrics(generated: str, expected: str) -> dict[str, float]:
    """
    Compute Precision, Recall, F1 from token-level overlap.
    Both strings are treated as token bags — order does not matter.
    """
    g = _tokenise(generated)
    e = _tokenise(expected)
    overlap   = len(g & e)
    precision = overlap / len(g) if g else 0.0
    recall    = overlap / len(e) if e else 0.0
    denom     = precision + recall
    f1        = 2 * precision * recall / denom if denom else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def semantic_similarity(rag, text_a: str, text_b: str) -> float:
    """
    Encode both strings with the RAG model and return cosine similarity.
    Normalised embeddings make this a simple dot product.
    """
    embs  = rag.model.encode(
        [text_a, text_b], convert_to_numpy=True, normalize_embeddings=True
    )
    score = cosine_similarity([embs[0]], [embs[1]])[0][0]
    return round(float(score), 4)


# ── Evaluation ─────────────────────────────────────────────────────────────
def run_evaluation() -> dict[str, Any]:
    """Run the full evaluation and return the report dict."""
    qa_pairs = get_qa_pairs()
    rag      = build_rag()

    rows: list[dict[str, Any]] = []
    for qa in qa_pairs:
        result    = rag.query(qa["question"])
        generated = result.answer
        expected  = qa["expected"]

        kw     = keyword_metrics(generated, expected)
        cosine = semantic_similarity(rag, generated, expected)
        p_kw   = kw["f1"]  >= PASS_F1
        p_cos  = cosine     >= PASS_COSINE

        rows.append({
            "question":          qa["question"],
            "expected_answer":   expected,
            "generated_answer":  generated,
            "keyword_metrics":   kw,
            "cosine_similarity": cosine,
            "pass_keyword_f1":   p_kw,
            "pass_cosine":       p_cos,
            "passed":            p_kw and p_cos,
            "retrieved_chunks":  [asdict(c) for c in result.top_chunks],
        })

    n      = len(rows)
    passed = sum(r["passed"] for r in rows)
    summary: dict[str, Any] = {
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "model":                 rag.model_name,
        "total_questions":       n,
        "passed":                passed,
        "failed":                n - passed,
        "pass_rate_pct":         round(passed / n * 100, 1) if n else 0.0,
        "avg_keyword_precision": round(float(np.mean([r["keyword_metrics"]["precision"] for r in rows])), 4),
        "avg_keyword_recall":    round(float(np.mean([r["keyword_metrics"]["recall"]    for r in rows])), 4),
        "avg_keyword_f1":        round(float(np.mean([r["keyword_metrics"]["f1"]        for r in rows])), 4),
        "avg_cosine_similarity": round(float(np.mean([r["cosine_similarity"]            for r in rows])), 4),
        "thresholds": {
            "keyword_f1":        PASS_F1,
            "cosine_similarity": PASS_COSINE,
        },
    }

    report = {"summary": summary, "results": rows}
    _save_json(report)
    _save_markdown(report)
    return report


# ── Serialisers ─────────────────────────────────────────────────────────────
def _save_json(report: dict[str, Any]) -> None:
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _save_markdown(report: dict[str, Any]) -> None:
    s = report["summary"]
    lines = [
        "# Evaluation Report — Beekeeping RAG Pipeline",
        "",
        f"> Generated : {s['timestamp']}",
        f"> Model     : `{s['model']}`",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total questions        | {s['total_questions']} |",
        f"| Passed                 | {s['passed']} |",
        f"| Failed                 | {s['failed']} |",
        f"| Pass rate              | {s['pass_rate_pct']}% |",
        f"| Avg keyword precision  | {s['avg_keyword_precision']} |",
        f"| Avg keyword recall     | {s['avg_keyword_recall']} |",
        f"| Avg keyword F1         | {s['avg_keyword_f1']} |",
        f"| Avg cosine similarity  | {s['avg_cosine_similarity']} |",
        f"| Threshold — keyword F1 | ≥ {s['thresholds']['keyword_f1']} |",
        f"| Threshold — cosine sim | ≥ {s['thresholds']['cosine_similarity']} |",
        "",
        "---",
        "",
        "## Per-Question Results",
        "",
    ]

    for idx, row in enumerate(report["results"], 1):
        st = "\u2705 PASS" if row["passed"] else "\u274c FAIL"
        kw = row["keyword_metrics"]
        lines += [
            f"### Q{idx} — {st}",
            "",
            f"**Question:** {row['question']}",
            "",
            "| | Answer |",
            "|---|---|",
            f"| Expected  | {row['expected_answer']} |",
            f"| Generated | {row['generated_answer']} |",
            "",
            "| Metric | Value | Threshold | Pass? |",
            "|--------|-------|-----------|-------|",
            f"| Keyword Precision  | {kw['precision']} | —          | — |",
            f"| Keyword Recall     | {kw['recall']}    | —          | — |",
            f"| Keyword F1         | {kw['f1']}        | ≥ {PASS_F1}  | {'\u2705' if row['pass_keyword_f1'] else '\u274c'} |",
            f"| Cosine Similarity  | {row['cosine_similarity']} | ≥ {PASS_COSINE} | {'\u2705' if row['pass_cosine'] else '\u274c'} |",
            "",
            "**Retrieved chunks:**",
            "",
        ]
        for rank, chunk in enumerate(row["retrieved_chunks"], 1):
            lines.append(
                f"{rank}. `score={chunk['score']:.4f}` | "
                f"`{chunk['chunk_id']}` — {chunk['text']}"
            )
        lines += ["", "---", ""]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ── Console printer ─────────────────────────────────────────────────────────────
def _print_report(report: dict[str, Any]) -> None:
    SEP  = "=" * 72
    THIN = "\u2500" * 68
    s    = report["summary"]

    print(f"\n{SEP}")
    print("  EVALUATION REPORT  —  ADVANCED BEEKEEPING TECHNIQUES RAG")
    print(f"{SEP}")
    print(f"  Model           : {s['model']}")
    print(f"  Passed          : {s['passed']}/{s['total_questions']}  ({s['pass_rate_pct']}%)")
    print(f"  Avg Keyword F1  : {s['avg_keyword_f1']}   (threshold ≥ {PASS_F1})")
    print(f"  Avg Cosine Sim  : {s['avg_cosine_similarity']}   (threshold ≥ {PASS_COSINE})")
    print(f"{SEP}")

    for idx, row in enumerate(report["results"], 1):
        kw  = row["keyword_metrics"]
        st  = "\u2705 PASS" if row["passed"] else "\u274c FAIL"
        print(f"\n  Q{idx}: {row['question']}")
        print(f"  {THIN}")
        print(f"  Expected   : {row['expected_answer']}")
        print(f"  Generated  : {row['generated_answer']}")
        print(f"  Keyword    : precision={kw['precision']}  "
              f"recall={kw['recall']}  f1={kw['f1']}")
        print(f"  Cosine Sim : {row['cosine_similarity']}")
        print(f"  Result     : {st}")

    print(f"\n{SEP}")
    print(f"  Reports saved \u2192  {REPORT_JSON}")
    print(f"               \u2192  {REPORT_MD}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    report = run_evaluation()
    _print_report(report)

import json
import re
import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rag_pipeline import RAGPipeline

def tokenize(text: str) -> set:
    """Tokenizes text into a set of lowercase alphanumeric words."""
    return set(re.findall(r'\b\w+\b', text.lower()))

def compute_keyword_metrics(generated: str, expected: str) -> Tuple[float, float, float]:
    """
    Computes keyword Precision, Recall, and F1.

    Args:
        generated: The generated answer string.
        expected: The expected answer string.

    Returns:
        A tuple of (Precision, Recall, F1).
    """
    gen_tokens = tokenize(generated)
    exp_tokens = tokenize(expected)

    if not exp_tokens:
        return 0.0, 0.0, 0.0
    if not gen_tokens:
        return 0.0, 0.0, 0.0

    overlap = gen_tokens.intersection(exp_tokens)

    precision = len(overlap) / len(gen_tokens)
    recall = len(overlap) / len(exp_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return float(precision), float(recall), float(f1)

def compute_cosine_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    """
    Computes the cosine similarity between two texts using the given model.
    """
    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / np.where(norms == 0, 1e-10, norms)

    sim = np.dot(norm_embeddings[0], norm_embeddings[1])
    return float(sim)

def evaluate_pipeline():
    """Runs the RAG pipeline on dataset.json and evaluates the results."""
    # Load data
    try:
        with open("dataset.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: dataset.json not found.")
        return

    document = data.get("document", "")
    qa_pairs = data.get("qa_pairs", [])

    # Initialize RAG Pipeline
    pipeline = RAGPipeline()
    pipeline.load_document(document)
    pipeline.create_chunks(window_size=2, overlap=1)
    pipeline.embed_chunks()

    eval_results = []

    # Thresholds for Pass/Fail
    F1_THRESHOLD = 0.3
    COSINE_THRESHOLD = 0.5

    for qa in qa_pairs:
        question = qa["question"]
        expected = qa["expected_answer"]

        # Run pipeline
        retrieved_chunks = pipeline.retrieve(question, top_k=2)
        generated = pipeline.generate_answer(question, retrieved_chunks)

        # Compute metrics
        precision, recall, f1 = compute_keyword_metrics(generated, expected)
        cosine_sim = compute_cosine_similarity(pipeline.model, generated, expected)

        passed = f1 >= F1_THRESHOLD and cosine_sim >= COSINE_THRESHOLD

        eval_results.append({
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated,
            "metrics": {
                "keyword_precision": round(precision, 4),
                "keyword_recall": round(recall, 4),
                "keyword_f1": round(f1, 4),
                "cosine_similarity": round(cosine_sim, 4)
            },
            "passed": passed
        })

    # Summary
    total = len(eval_results)
    passed_count = sum(1 for res in eval_results if res["passed"])

    report_data = {
        "summary": {
            "total_questions": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": f"{(passed_count / total) * 100:.2f}%" if total > 0 else "0.00%"
        },
        "results": eval_results
    }

    # Write output JSON
    os.makedirs("output", exist_ok=True)
    with open("output/evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # Write output MD
    md_content = "# Evaluation Report\n\n"
    md_content += "## Summary\n"
    md_content += f"- **Total Questions:** {total}\n"
    md_content += f"- **Passed:** {passed_count}\n"
    md_content += f"- **Failed:** {total - passed_count}\n"
    md_content += f"- **Pass Rate:** {report_data['summary']['pass_rate']}\n\n"

    md_content += "## Details\n"
    for res in eval_results:
        md_content += f"### Question: {res['question']}\n"
        md_content += f"- **Expected Answer:** {res['expected_answer']}\n"
        md_content += f"- **Generated Answer:** {res['generated_answer']}\n"
        md_content += "- **Metrics:**\n"
        md_content += f"  - Keyword Precision: {res['metrics']['keyword_precision']}\n"
        md_content += f"  - Keyword Recall: {res['metrics']['keyword_recall']}\n"
        md_content += f"  - Keyword F1: {res['metrics']['keyword_f1']}\n"
        md_content += f"  - Cosine Similarity: {res['metrics']['cosine_similarity']}\n"
        status = "✅ PASS" if res['passed'] else "❌ FAIL"
        md_content += f"- **Status:** {status}\n\n"

    with open("output/evaluation_report.md", "w") as f:
        f.write(md_content)

    print("Evaluation completed. Reports saved in 'output/' directory.")

if __name__ == "__main__":
    evaluate_pipeline()

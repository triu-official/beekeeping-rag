# Evaluation Report — Beekeeping RAG Pipeline

> Generated : 2026-06-26T08:16:00+00:00
> Model     : `all-MiniLM-L6-v2`

---

## Summary

| Metric | Value |
|--------|-------|
| Total questions        | 3 |
| Passed                 | 3 |
| Failed                 | 0 |
| Pass rate              | 100.0% |
| Avg keyword precision  | 0.3929 |
| Avg keyword recall     | 0.9167 |
| Avg keyword F1         | 0.5333 |
| Avg cosine similarity  | 0.8838 |
| Threshold — keyword F1 | ≥ 0.3 |
| Threshold — cosine sim | ≥ 0.6 |

---

## Per-Question Results

### Q1 — ✅ PASS

**Question:** What is the minimum internal temperature for a Langstroth Hive in winter?

| | Answer |
|---|---|
| Expected  | Above 40 degrees Fahrenheit. |
| Generated | The internal hive temperature must be maintained above 40 degrees Fahrenheit to prevent the colony from freezing. |

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Keyword Precision  | 0.25   | —         | —     |
| Keyword Recall     | 1.0    | —         | —     |
| Keyword F1         | 0.4    | ≥ 0.3     | ✅    |
| Cosine Similarity  | 0.8721 | ≥ 0.6     | ✅    |

**Retrieved chunks:**

1. `score=0.6142` | `doc_001_c001` — Winterization of Langstroth Hives requires strict temperature management. The internal hive temperature must be maintained above 40 degrees Fahrenheit to prevent the colony from freezing.
2. `score=0.2894` | `doc_001_c002` — The internal hive temperature must be maintained above 40 degrees Fahrenheit to prevent the colony from freezing. Beekeepers often use insulated wraps and moisture quilt boxes to control condensation.

---

### Q2 — ✅ PASS

**Question:** Why are entrance reducers used?

| | Answer |
|---|---|
| Expected  | To prevent field mice from entering. |
| Generated | Entrance reducers are placed to prevent field mice from entering during the colder months. |

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Keyword Precision  | 0.4286 | —         | —     |
| Keyword Recall     | 1.0    | —         | —     |
| Keyword F1         | 0.6    | ≥ 0.3     | ✅    |
| Cosine Similarity  | 0.895  | ≥ 0.6     | ✅    |

**Retrieved chunks:**

1. `score=0.5873` | `doc_001_c003` — Beekeepers often use insulated wraps and moisture quilt boxes to control condensation. Entrance reducers are placed to prevent field mice from entering during the colder months.
2. `score=0.3011` | `doc_001_c004` — Entrance reducers are placed to prevent field mice from entering during the colder months.

---

### Q3 — ✅ PASS

**Question:** How do beekeepers control condensation?

| | Answer |
|---|---|
| Expected  | By using insulated wraps and moisture quilt boxes. |
| Generated | Beekeepers often use insulated wraps and moisture quilt boxes to control condensation. |

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Keyword Precision  | 0.5    | —         | —     |
| Keyword Recall     | 0.75   | —         | —     |
| Keyword F1         | 0.6    | ≥ 0.3     | ✅    |
| Cosine Similarity  | 0.8843 | ≥ 0.6     | ✅    |

**Retrieved chunks:**

1. `score=0.6217` | `doc_001_c003` — Beekeepers often use insulated wraps and moisture quilt boxes to control condensation. Entrance reducers are placed to prevent field mice from entering during the colder months.
2. `score=0.4498` | `doc_001_c002` — The internal hive temperature must be maintained above 40 degrees Fahrenheit to prevent the colony from freezing. Beekeepers often use insulated wraps and moisture quilt boxes to control condensation.

---

# 🔎 BM25 — Sparse Retrieval

> **BM25** (Best Matching 25) is a classic **keyword-based** retrieval algorithm — the backbone of search engines before dense embeddings took over.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Rank documents by relevance to a query using BM25 |

---

## 🧠 How BM25 Works

BM25 scores each document by how well its **keywords** match the query, factoring in:
- **Term frequency** — how often the query word appears in the document
- **Inverse document frequency** — rarer words get higher weight
- **Document length normalization** — long documents don't get unfair advantage

```python
from rank_bm25 import BM25Okapi

corpus = ["papers are made using trees", "reading books help improve your speaking skills", ...]
tokenized = [doc.lower().split() for doc in corpus]

bm25 = BM25Okapi(tokenized)
scores = bm25.get_scores("how to improve my speaking skills".lower().split())
```

---

## 🛠️ Requirements

```bash
pip install rank-bm25
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 🆚 BM25 vs Dense Retrieval

| | BM25 | Dense Retrieval |
|---|------|-----------------|
| Matching | Exact keywords | Semantic meaning |
| Speed | Very fast | Slower (embedding computation) |
| Handles synonyms | No | Yes |
| Best for | Keyword-heavy queries | Natural language queries |

---

## 🔁 Related

- [`../Dense_Retrival/`](../Dense_Retrival/) — dense retrieval with sentence embeddings
- [`../RAG/`](../RAG/) — combine retrieval with LLM generation

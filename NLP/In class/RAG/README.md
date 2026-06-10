# 🔗 RAG — Retrieval-Augmented Generation

> **RAG** combines a **retriever** (find relevant documents) with a **generator** (LLM produces an answer grounded in those documents). The result: factual, context-aware answers without fine-tuning.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `RAG.ipynb` | Core RAG pipeline — embed docs, retrieve top-k, generate with Phi-2 |
| `RAG2.ipynb` | Extended RAG experiments and variations |
| `RAGExercises.ipynb` | Hands-on exercises to build your own RAG system |
| `test.pdf` | Sample PDF used as a knowledge source |

---

## 🧠 The RAG Pipeline

```
Query → Embed → Retrieve top-k docs → Build prompt with context → LLM generates answer
```

1. **Embed** documents with `SentenceTransformer("all-MiniLM-L6-v2")`
2. **Retrieve** the most relevant chunks via cosine similarity
3. **Build a prompt** — inject retrieved context + the user's question
4. **Generate** an answer with an LLM (e.g. `microsoft/Phi-2`)

---

## ⚡ Key Code

```python
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Retrieve
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# ... embed docs, compute cosine similarity, pick top-k ...

# Generate
generator = pipeline("text-generation", model="microsoft/Phi-2")
answer = generator(f"Context: {context}\nQuestion: {query}\nAnswer:", max_new_tokens=80)
```

---

## 🛠️ Requirements

```bash
pip install sentence-transformers transformers torch scikit-learn
```

---

## ▶️ Run

```bash
jupyter notebook RAG.ipynb
jupyter notebook RAG2.ipynb
jupyter notebook RAGExercises.ipynb
```

---

## 💡 Why RAG?

| Problem | RAG solves it by… |
|---------|-------------------|
| LLMs hallucinate | Grounding answers in retrieved documents |
| Knowledge is outdated | Retrieving from an up-to-date corpus |
| Fine-tuning is expensive | No retraining needed — just update the document store |

---

## 🔁 Related

- [`../BM25/`](../BM25/) — sparse retrieval (keyword matching)
- [`../Dense_Retrival/`](../Dense_Retrival/) — dense retrieval (embeddings)
- [`../Agents/`](../Agents/) — agents that can use retrieval as a tool

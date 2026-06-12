# 🧲 Dense Retrieval

> Find relevant documents by **meaning**, not just keywords — using **sentence embeddings** and cosine similarity.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Dense retrieval on a small corpus + PDF chunking at scale |
| `test.pdf` | Sample PDF for chunking experiments |
| `Natural-Language-Processing-Python (1).pdf` | Reference book (used as retrieval corpus) |
| `learning-langchain-for-true-epub-9781098167288.pdf` | Reference book (used as retrieval corpus) |

---

## 🧠 How It Works

1. **Encode** documents and query into dense vectors using a sentence-transformer model
2. **Compare** vectors with cosine similarity
3. **Rank** documents by similarity score

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(docs)
query_emb = model.encode(query)
similarities = cosine_similarity(query_emb.reshape(1, -1), doc_emb)[0]
```

### PDF Chunking Pipeline
The notebook also demonstrates retrieval over **real PDF books**:
1. Extract text from PDFs with `pypdf`
2. Split into fixed-size chunks (300 chars)
3. Embed all chunks
4. Retrieve top-k chunks for any query

---

## 🛠️ Requirements

```bash
pip install sentence-transformers scikit-learn pypdf
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 🔁 Related

- [`../BM25/`](../BM25/) — sparse (keyword) retrieval for comparison
- [`../RAG/`](../RAG/) — feed retrieved chunks to an LLM for answer generation

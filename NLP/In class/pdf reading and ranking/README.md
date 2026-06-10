# 📄 PDF Reading and Ranking

> Extract text from PDF files and **rank pages/chunks** by relevance to a query using dense retrieval.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `code.ipynb` | Extract text from PDFs, embed chunks, rank by query similarity |
| `test.pdf` | Sample PDF for testing |
| `Natural-Language-Processing-Python (1).pdf` | NLP reference book (used as corpus) |
| `learning-langchain-for-true-epub-9781098167288.pdf` | LangChain reference (used as corpus) |

---

## 🧠 Pipeline

1. **Extract** text from PDFs using `pypdf`
2. **Chunk** the text into fixed-size segments
3. **Embed** each chunk with `SentenceTransformer`
4. **Query** — embed the user's question and rank chunks by cosine similarity
5. **Display** the top-k most relevant chunks with their source PDF and score

---

## 🛠️ Requirements

```bash
pip install pypdf sentence-transformers scikit-learn
```

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 🔁 Related

- [`../Dense_Retrival/`](../Dense_Retrival/) — same dense retrieval technique on smaller corpora
- [`../RAG/`](../RAG/) — feed the retrieved chunks into an LLM for answer generation

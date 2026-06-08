# 🔍 String Similarity (Distances)

> How "similar" are two pieces of text?
> Different similarity measures answer that question at **different levels** — characters, words or vectors.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `p1.ipynb` | Hands-on with edit distance, Jaccard & cosine similarity |

---

## 🧠 Three Levels of Similarity

| Measure | Operates on | Interpretation |
|---------|-------------|----------------|
| **Levenshtein (Edit) distance** | Characters | # of insert / delete / substitute operations to turn A into B. Lower = more similar. |
| **Jaccard similarity** | Word sets | `|A ∩ B| / |A ∪ B|`. Range 0–1, higher = more overlap. |
| **Cosine similarity** | TF-IDF / embedding vectors | `cos(θ)` between vectors. Range 0–1, higher = more aligned. |

---

## 🎯 What You'll Learn

1. Compute **edit distance** with `python-Levenshtein` (or pure Python)
2. Compute **Jaccard** on the set of tokens of two sentences
3. Compute **cosine similarity** between TF-IDF vectors
4. Build a **spelling corrector** that picks the dictionary word with the smallest edit distance
5. Build a **near-duplicate detector** using cosine similarity

---

## 🛠️ Requirements

```bash
pip install scikit-learn nltk python-Levenshtein
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 💡 When to Use What

| You want to… | Use |
|--------------|-----|
| Correct typos | **Edit distance** |
| Compare short word lists | **Jaccard** |
| Compare documents | **Cosine on TF-IDF** |
| Find semantically similar sentences | **Cosine on embeddings** |

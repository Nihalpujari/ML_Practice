# 📊 Module 4 — Feature Extraction (BoW)

> A focused, classroom-style notebook on **Bag-of-Words**, dated **26 May**.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `26.05 BoW.ipynb` | Step-by-step Bag-of-Words walkthrough |

---

## 🧠 What's Inside

A pedagogical walkthrough of **Bag-of-Words** representation:

1. Pick a small corpus of sentences
2. Preprocess (lowercase, tokenize, remove stop words & punctuation)
3. Fit `CountVectorizer` to build the vocabulary
4. Transform documents into count vectors
5. Inspect the resulting **document-term matrix**
6. Optionally extend to **n-grams** (bigrams capture some word order)

---

## 🛠️ Requirements

```bash
pip install scikit-learn nltk pandas
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## ▶️ Run

```bash
jupyter notebook "26.05 BoW.ipynb"
```

---

## 💡 Note

This notebook also appears under [`In Class/module 4 feature extraction/`](../In%20Class/module%204%20feature%20extraction/) — the duplication mirrors the original course folder structure.

For deeper feature-extraction examples (TF-IDF, embeddings), see [`../Feature extraction/`](../Feature%20extraction/).

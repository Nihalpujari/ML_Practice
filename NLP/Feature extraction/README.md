# 🔣 Feature Extraction

> Models work with numbers, not strings.
> **Feature extraction** turns clean text into numeric vectors so ML algorithms can use it.

---

## 📂 Contents

| File | Topic | Tools |
|------|-------|-------|
| `p6.py` | **Bag of Words** + bar chart | `CountVectorizer`, matplotlib |
| `p7.ipynb` | **TF-IDF** + heatmap | `TfidfVectorizer`, seaborn |
| `p8.ipynb` | **GloVe word embeddings** + PCA | Gensim, scikit-learn |

---

## 🧠 The Three Levels of Text Representation

```
Text  →  Counts  →  Weighted Counts  →  Dense Vectors
(words)   (BoW)        (TF-IDF)         (Embeddings)
```

Each step captures **more semantic information** than the last.

---

## 1️⃣ `p6.py` — Bag of Words

Encodes each document as a **vector of word counts** over a fixed vocabulary.
Ignores grammar and word order — just frequencies.

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_reviews)
```

Demo dataset — 3 product reviews:
- *"The product is fantastic! It works like a charm."*
- *"I hated the product. It broke after one use."*
- *"Product was okay, not the best, but fine overall."*

📊 Output: a bar chart of word frequencies across the corpus.

---

## 2️⃣ `p7.ipynb` — TF-IDF

**TF-IDF (Term Frequency × Inverse Document Frequency)** improves on BoW by:
- 🟢 Rewarding words that are **frequent in one document**
- 🔴 Penalizing words **common across all documents** (so "the", "is" are downweighted)

```
TF-IDF(t, d) = TF(t, d) × log(N / df(t))
```

📊 Output: a seaborn **heatmap** of TF-IDF scores (words × reviews).

---

## 3️⃣ `p8.ipynb` — GloVe Word Embeddings + PCA

Pre-trained `glove-wiki-gigaword-50` vectors loaded via Gensim. Each word becomes a **dense 50-D vector** that captures semantic meaning.

```python
import gensim.downloader as api
model = api.load('glove-wiki-gigaword-50')

model['movie']                      # 50-D vector
model.similarity('movie', 'film')   # ~0.83
model.most_similar('king')          # ['queen', 'prince', ...]
```

📊 Output: PCA-projected 2-D scatter plot — words like `movie/film` and `cat/dog` should cluster together.

---

## 🆚 BoW vs TF-IDF vs Embeddings

| | BoW | TF-IDF | Embeddings |
|---|---|---|---|
| Captures word order | ❌ | ❌ | Partially |
| Penalizes common words | ❌ | ✅ | n/a |
| Sparse / Dense | Sparse | Sparse | Dense |
| Vector size | = vocab | = vocab | Fixed (50/100/300…) |
| Semantic meaning | ❌ | ❌ | ✅ |

---

## 🛠️ Requirements

```bash
pip install scikit-learn nltk gensim matplotlib seaborn pandas numpy
```

`glove-wiki-gigaword-50` is downloaded automatically on first run (~70 MB).

---

## ▶️ Run

```bash
python p6.py
jupyter notebook p7.ipynb
jupyter notebook p8.ipynb
```

---

## 💡 Tips

- **Always preprocess first** (lowercasing, stop-word removal) — otherwise "The" and "the" become different vocab entries.
- Try **`ngram_range=(1, 2)`** to include bigrams ("not good" ≠ "not" + "good").
- For modern NLP, use **sentence embeddings** (e.g. `sentence-transformers`) for even richer semantic vectors.

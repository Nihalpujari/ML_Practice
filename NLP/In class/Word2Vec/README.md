# 🔤 Word2Vec — Learn Word Vectors from Scratch

> **Word2Vec** learns dense vector representations of words by predicting nearby words (or vice-versa) from a large text corpus.
> Output: `king − man + woman ≈ queen` style algebra.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Train a small Word2Vec model with Gensim |

---

## 🧠 The Two Word2Vec Architectures

| Architecture | What it predicts |
|--------------|------------------|
| **CBOW** (Continuous Bag of Words) | Given context words, predict the center word |
| **Skip-gram** | Given the center word, predict the context words |

Both result in a learned **embedding matrix** — every word gets a dense vector.

---

## ⚡ Demo

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ...
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,            # 1 = Skip-gram, 0 = CBOW
    workers=4
)

model.wv["cat"]                    # 100-D vector
model.wv.most_similar("cat", topn=5)
```

---

## 🛠️ Requirements

```bash
pip install gensim nltk
python -c "import nltk; nltk.download('punkt')"
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 💡 Tips

- **Big corpus or pre-trained.** For small projects, use pre-trained embeddings (`gensim.downloader.load('glove-wiki-gigaword-100')`) instead of training from scratch.
- **`window`** controls how far context can reach — larger window = more "topical" similarity.
- For **modern semantic search**, look at `sentence-transformers` (sentence-level embeddings).

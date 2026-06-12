# 🧽 Text Cleaning Techniques

> After tokenization, text needs to be **cleaned** before models can make sense of it.
> Less noise = better features = better models.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `p3.py` | Stop-word & punctuation removal with NLTK |
| `p3.3.py` | Quick spaCy lemmatization experiment |
| `p4.py` | Stemming (Porter) vs Lemmatization (WordNet) side-by-side |
| `p5.py` | Full preprocessing pipeline on a travel review |
| `re p1.ipynb` | Regular expressions — basics |
| `re p2.ipynb` | Regular expressions — more advanced patterns |
| `stemmer_spacy.ipynb` | spaCy-based lemmatization in a notebook |

---

## 🧱 The Standard Cleaning Pipeline

```
raw text
   ↓ lowercase
   ↓ tokenize
   ↓ remove stop words
   ↓ remove punctuation
   ↓ stem or lemmatize
clean tokens
```

---

## 🔍 What Each Script Does

### `p3.py` — Stop words & punctuation
Removes English stop words (`the`, `is`, `at`…) and punctuation (`.,!?`) from a tokenized sentence.

```python
filtered = [w for w in tokens if w.lower() not in stop_words]
clean    = [w for w in filtered if w not in string.punctuation]
```

### `p4.py` — Stemming vs Lemmatization
Side-by-side comparison on `["running", "ran", "bats", "organizations", …]`:

| Method | Tool | Real word? |
|--------|------|-----------|
| Stemming | `PorterStemmer` | ❌ (`organ` from "organizations") |
| Lemmatization | `WordNetLemmatizer` | ✅ (`organization`) |

### `p5.py` — Full pipeline
Applies lowercasing → tokenization → stop-word removal → punctuation removal on a casual travel review.

### `re p1.ipynb` / `re p2.ipynb` — Regular expressions
Hands-on Python `re` module patterns — perfect for extracting emails, URLs, hashtags, numbers, etc., before feeding text to ML models.

### `stemmer_spacy.ipynb` — spaCy lemmatizer
Notebook showcasing spaCy's POS-aware lemmatizer (more accurate than NLTK's because it knows the part of speech).

### `p3.3.py` — spaCy quick test
⚠️ Contains intentional typos / errors — useful as a "debug me" exercise.

---

## 🛠️ Requirements

```bash
pip install nltk spacy
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

---

## ▶️ Run

```bash
python p3.py
python p4.py
python p5.py
jupyter notebook "re p1.ipynb"
```

---

## 💡 Tips

- **Don't always remove stop words.** "not", "never", "no" are critical for sentiment analysis.
- **Lemmatization > stemming** when you need real words (chatbots, generation).
- **Stemming > lemmatization** when speed matters (search indexes, big-data pipelines).
- Build cleaning as a **single function** so train and inference use exactly the same logic.

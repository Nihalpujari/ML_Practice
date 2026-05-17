# 🧠 NLP — Natural Language Processing with Python

> A hands-on collection of NLP fundamentals implemented using **NLTK**, **Scikit-learn**, and **Gensim**.
> Covers the complete NLP journey — from raw text tokenization to word embeddings.

---

## 📁 Folder Structure

```
NLP/
│
├── 📂 Tokenization/
│   ├── p1.py              → Word Tokenization
│   └── p2.py              → Sentence Tokenization
│
├── 📂 cleaning and its techniques/
│   ├── p3.py              → Stop Words & Punctuation Removal
│   ├── p4.py              → Stemming & Lemmatization
│   └── p5.py              → Full Text Preprocessing Pipeline
│
└── 📂 Feature extraction/
    ├── p6.py              → Bag of Words (BoW) + Bar Chart
    ├── p7.ipynb           → TF-IDF Vectorization + Heatmap
    └── p8.ipynb           → Word Embeddings (GloVe) + PCA
```

---

## 📂 Tokenization

Tokenization is the first step in any NLP pipeline — splitting raw text into smaller meaningful units.

### `p1.py` — Word Tokenization
Splits a paragraph into individual **words** using `nltk.word_tokenize()`.

- **Library:** `nltk`
- **Input:** A raw text paragraph about the stock market
- **Output:** A list of word tokens
- **Key function:**
```python
tokens = nltk.word_tokenize(text)
```

---

### `p2.py` — Sentence Tokenization
Splits a paragraph into individual **sentences** using `nltk.sent_tokenize()`.

- **Library:** `nltk`
- **Input:** Same stock market paragraph
- **Output:** A list of sentences
- **Key function:**
```python
sentences = nltk.sent_tokenize(text)
```

---

## 📂 Cleaning and Its Techniques

After tokenization, text must be cleaned to remove noise before feeding it into a model.

### `p3.py` — Stop Words & Punctuation Removal
Filters out common English stop words (e.g., *"is", "the", "and"*) and punctuation from tokenized text.

- **Library:** `nltk`, `string`
- **Input:** A sentence about NLP text preprocessing
- **Output:** A clean list of meaningful words only
- **Key steps:**
```python
filtered = [w for w in tokens if w not in stop_words]
clean    = [w for w in filtered if w not in string.punctuation]
```

---

### `p4.py` — Stemming & Lemmatization
Compares two word normalization techniques side by side:

| Technique | Tool | Example |
|-----------|------|---------|
| **Stemming** | `PorterStemmer` | "running" → "run" |
| **Lemmatization** | `WordNetLemmatizer` | "bats" → "bat" |

- **Library:** `nltk`
- **Input:** A list of words — `["running", "ran", "bats", "organizations", ...]`
- **Output:** Both stemmed and lemmatized versions printed side by side

---

### `p5.py` — Full Text Preprocessing Pipeline
A complete, real-world text cleaning pipeline applied to a travel/airline review. Chains together:

1. **Lowercasing** — normalize case
2. **Tokenization** — split into words
3. **Stop word removal** — remove filler words
4. **Punctuation removal** — strip symbols

- **Library:** `nltk`, `string`
- **Input:** A casual review with mixed case and informal language
- **Output:** A clean list of meaningful tokens ready for modeling

---

## 📂 Feature Extraction

Feature extraction converts cleaned text into **numerical representations** that machine learning models can understand.

### `p6.py` — Bag of Words (BoW) + Bar Chart
Implements the **Bag of Words** model using `CountVectorizer` on three product reviews. Visualizes word frequencies as a **bar chart**.

- **Library:** `sklearn`, `nltk`, `matplotlib`, `numpy`
- **Input:** 3 product reviews (positive, negative, neutral)
- **Output:** Word frequency bar chart
- **Key concept:** Each document is represented as a vector of word counts, ignoring grammar and word order
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_reviews)
```

---

### `p7.ipynb` — TF-IDF Vectorization + Heatmap 📓
Implements **TF-IDF (Term Frequency–Inverse Document Frequency)** on the same product reviews. Unlike BoW, TF-IDF rewards words that are important in a document but rare across all documents. Visualizes the full TF-IDF matrix as a **seaborn heatmap**.

- **Library:** `sklearn`, `nltk`, `pandas`, `seaborn`, `matplotlib`
- **Input:** 3 product reviews
- **Output:** TF-IDF heatmap (words × reviews)
- **Key concept:** Penalizes very common words and highlights unique/important terms
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_reviews)
```

---

### `p8.ipynb` — Word Embeddings (GloVe) + PCA Visualization 📓
Loads pre-trained **GloVe word vectors** (50-dimensional) via Gensim and explores semantic word relationships. Reduces dimensions using **PCA** and plots words in 2D space.

- **Library:** `gensim`, `sklearn`, `matplotlib`
- **Model:** Pre-trained `glove-wiki-gigaword-50`
- **Words explored:** `movie, film, bus, car, train, actor, dog, cat, king, queen`
- **Output:** 2D PCA scatter plot showing semantic clusters
- **Key concepts explored:**
  - Word vector lookup: `model['movie']`
  - Word similarity score: `model.similarity('movie', 'film')`
  - Top similar words: `model.most_similar('movie', topn=10)`
  - Dimensionality reduction with PCA for visualization

---

## 🛠️ Requirements

```bash
pip install nltk scikit-learn gensim matplotlib seaborn pandas numpy
```

Also download the required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🚀 How to Run

```bash
# Run any Python script
python "NLP/Tokenization/p1.py"

# Open notebooks in Jupyter
jupyter notebook "NLP/Feature extraction/p7.ipynb"
```

---

## 📌 Topics at a Glance

| File | Folder | Topic |
|------|--------|-------|
| `p1.py` | Tokenization | Word Tokenization |
| `p2.py` | Tokenization | Sentence Tokenization |
| `p3.py` | Cleaning and its techniques | Stop Words & Punctuation Removal |
| `p4.py` | Cleaning and its techniques | Stemming & Lemmatization |
| `p5.py` | Cleaning and its techniques | Full Preprocessing Pipeline |
| `p6.py` | Feature extraction | Bag of Words (BoW) + Bar Chart |
| `p7.ipynb` | Feature extraction | TF-IDF Vectorization + Heatmap |
| `p8.ipynb` | Feature extraction | Word Embeddings (GloVe) + PCA |

# Movie Recommendation System using TF-IDF & Cosine Similarity

A content-based search and recommendation system that matches user queries to movies using natural language processing techniques.

---

## Overview

This project implements a lightweight recommendation engine from scratch using TF-IDF vectorization and cosine similarity. Given a free-text query, the system returns the top 3 most relevant movies from a curated dataset of 10 sci-fi films.

---

## How It Works

1. **Dataset** — A hand-crafted dataset of 10 movies, each with a title and a short description.
2. **Text Preprocessing** — Each description (and the user query) is cleaned through a pipeline:
   - Lowercasing
   - Punctuation removal
   - Stopword filtering
   - Lemmatization (via NLTK's `WordNetLemmatizer`)
3. **TF-IDF Vectorization** — Preprocessed descriptions are converted into numerical vectors using `TfidfVectorizer` from scikit-learn, producing a `(10, 141)` feature matrix.
4. **Query Processing** — The user's query goes through the same preprocessing pipeline and is transformed using the fitted vectorizer.
5. **Cosine Similarity** — Similarity scores are computed between the query vector and all document vectors, and the top-k results are returned.

---

## Project Structure

```
search_recommendation_system.ipynb   # Main notebook
README.md                            # This file
```

---

## Requirements

```
numpy
pandas
scikit-learn
nltk
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn nltk
```

You will also need the following NLTK data packages. Download them once in Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## Usage

Run the notebook cells in order. To query the system, call the `search()` function with any natural language string:

```python
query = "robots and artificial intelligence"
results = search(query, top_k=3)
print_results(query, results)
```

**Example output:**
```
Query: "Artificial Intelligence"
-----------------------------------------------------------------
1. Ex Machina
   Score      : 0.4718
   Description: A programmer is invited to test an advanced artificial intelligence robot...

2. Blade Runner
   Score      : 0.1530
   Description: A detective hunts down artificial humanoids called replicants...

3. Avatar
   Score      : 0.1489
   Description: A paraplegic marine travels to an alien moon...
```

---

## Notes & Limitations

- **Small dataset** — Only 10 movies are included. Queries unrelated to the dataset (e.g. *"Happy new year"*) may return low-confidence results with a similarity score of `0.0`.
- **Vocabulary-bound** — TF-IDF relies on exact or lemmatized word overlap. It has no semantic understanding, so synonyms or paraphrases may not match well. Consider upgrading to sentence embeddings (e.g. `sentence-transformers`) for better semantic recall.
- **Static vectorizer** — The vectorizer is fit on the dataset at load time. Adding new documents requires re-fitting.

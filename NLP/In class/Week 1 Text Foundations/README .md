# NLP Lab — Week 1: Text Foundations

A hands-on introductory lab covering the core building blocks of Natural Language Processing (NLP). By the end of this lab, you will have implemented and compared the most fundamental text preprocessing and representation techniques used in modern NLP pipelines.

---

## Lab Roadmap

| # | Topic |
|---|-------|
| 0 | Setup & Library Check |
| 1 | Tokenization |
| 2 | Stemming & Lemmatization |
| 3 | Stop Words & Text Cleaning |
| 4 | Bag of Words & N-grams |
| 5 | TF-IDF |
| 6 | String Similarity |

---

## Prerequisites

### Required Libraries

The notebook installs all dependencies automatically in **Cell A** (Section 0). The following packages are used:

| Library | Purpose |
|---------|---------|
| `nltk` | Tokenization, stemming, lemmatization, stopwords |
| `spacy` | Rule + ML hybrid NLP pipeline |
| `scikit-learn` | CountVectorizer, TfidfVectorizer, cosine similarity |
| `transformers` | GPT-2 / BERT sub-word tokenizers (Hugging Face) |
| `Levenshtein` | Fast edit-distance computation |
| `pandas` | Tabular display of results |
| `numpy` | Numerical operations on vectors |
| `tabulate` | Clean text table formatting |

### spaCy Model

The notebook automatically downloads `en_core_web_sm` (the small English pipeline) if it is not already installed.

---

## Lab Philosophy

Each section contains **three layers**:

- **Concept** — what the technique is and why it matters
- **Starter code** — heavily commented, read before running
- **Your turn** — fill-in cells with hints and reflection questions

### Difficulty Key

| Symbol | Level |
|--------|-------|
| 🟢 | Beginner-friendly |
| 🟡 | Intermediate |
| 🔴 | Challenging / Exciting |

Cells marked `✍️` require your input — either code or a written response.

---

## Section Summaries

### Section 0 — Setup
Run **Cell A** to install packages and **Cell B** to import them and download NLTK resources. On first run, Cell B may take a few minutes.

### Section 1 — Tokenization
Compare five tokenization strategies on real-world text including contractions, URLs, hashtags, and numeric formats:

- **Whitespace** — `str.split()`
- **Regex** — hand-written pattern
- **NLTK** — Penn Treebank-trained tokenizer
- **spaCy** — rule + ML hybrid
- **GPT-2 BPE** — learned sub-word merge rules

**Exercises:**
- 1.1 🟢 Sentence tokenization with `sent_tokenize`
- 1.2 🟡 Social-media regex tokenizer (hashtags, mentions, URLs)
- 1.3 🔴 BERT WordPiece vs GPT-2 BPE on medical text

### Section 2 — Stemming & Lemmatization
Normalize words to a common base form using four approaches:

| Method | Output always a real word? |
|--------|---------------------------|
| Porter Stemmer | No |
| Snowball Stemmer | No |
| WordNet Lemmatizer | Yes |
| spaCy Lemmatizer | Yes |

**Rule of thumb:** Use stemming for speed (e.g., search indexes); use lemmatization when linguistic accuracy matters (e.g., QA, generation).

**Exercises:**
- 2.1 🟢 Normalize a product review; compare unique form counts
- 2.2 🟡 spaCy POS-aware lemmatization vs NLTK without POS hints

### Section 3 — Stop Words & Text Cleaning
A reusable 5-step cleaning pipeline: lowercase → remove URLs → remove punctuation → tokenize → remove stop words.

> ⚠️ Removing stop words is **not always correct** — words like "not" and "never" are critical for sentiment analysis.

**Exercises:**
- 3.1 🟢 Apply the pipeline to a review corpus; measure token reduction
- 3.2 🟡 Build a domain-specific stop word list for medical notes

### Section 4 — Bag of Words & N-grams
Represent documents as count vectors over a fixed vocabulary. Word order is discarded; only frequencies matter.

**Key point:** Bigrams like "not good" are very different from the individual unigrams "not" and "good".

**Exercises:**
- 4.1 🟢 Bigram CountVectorizer for sentiment documents
- 4.2 🟡 Term frequency analysis — top terms, universal terms, unique terms

### Section 5 — TF-IDF
Weight terms by how **frequent they are in a document** but **rare across the corpus** — exactly the words that identify a document's unique topic.

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Exercise:**
- 5.1 🟡 Mini search engine using cosine similarity on TF-IDF vectors

### Section 6 — String Similarity
Compare three similarity measures operating at different levels:

| Measure | Level | Interpretation |
|---------|-------|----------------|
| Edit Distance (Levenshtein) | Character | Lower = more similar |
| Jaccard | Word set | 0–1, higher = more overlap |
| Cosine | TF-IDF vector | 0–1, higher = more similar |

**Exercises:**
- 6.1 🟢 Spelling corrector using edit distance
- 6.2 🔴 Near-duplicate review detector (bonus)

---

## Key Takeaways

| Topic | Remember |
|-------|---------|
| Tokenization | No single "best" tokenizer — match the method to your task |
| Stemming vs Lemmatization | Stemming = fast & approximate; lemmatization = linguistically accurate |
| Stop words | Removing them is NOT always correct (especially for sentiment!) |
| Bag of Words | Simple but effective; bigrams capture some word-order information |
| TF-IDF | Rewards locally-frequent AND globally-rare words |
| Similarity | Edit = character-level; Jaccard = set overlap; Cosine = vector angle |

---

## Further Reading

- [NLTK Book (free)](https://www.nltk.org/book/) — Chapters 1–3
- [spaCy 101](https://spacy.io/usage/spacy-101)
- [Hugging Face Tokenizer docs](https://huggingface.co/docs/tokenizers)
- [scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Jurafsky & Martin, *Speech and Language Processing*, Ch. 2 (free PDF)](https://web.stanford.edu/~jurafsky/slp3/)

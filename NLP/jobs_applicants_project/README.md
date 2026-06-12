# 💼 Job Applicants — NLP Project

> Apply NLP techniques to a real job-applicant / job-description matching problem.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `applicationproject.ipynb` | First version of the analysis |
| `applicationproject1.ipynb` | Iteration / improved version |

---

## 🎯 Project Idea

Given two pools of free text — **job descriptions** and **applicant profiles / resumes** — use NLP to:

1. **Preprocess** both corpora (clean, tokenize, lemmatize)
2. **Vectorize** them (TF-IDF, BoW, or embeddings)
3. **Compute similarity** between each applicant and each job
4. **Rank** applicants per job (or jobs per applicant)
5. Optionally classify resumes into role categories

---

## 🛠️ Tools Likely Used

- `nltk` / `spacy` — preprocessing
- `scikit-learn` — `TfidfVectorizer`, `cosine_similarity`
- `pandas` — tabular data handling
- (Optionally) sentence embeddings for richer semantic match

---

## ▶️ Run

```bash
jupyter notebook applicationproject.ipynb
jupyter notebook applicationproject1.ipynb
```

---

## 💡 Possible Extensions

- Add **NER** (see [`../named entity recognition and extraction/`](../named%20entity%20recognition%20and%20extraction/)) to extract skills, locations, companies
- Replace TF-IDF with **sentence-transformer embeddings** for true semantic matching
- Wrap the ranker in a **Streamlit** app for an interactive demo

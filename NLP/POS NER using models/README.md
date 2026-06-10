# 🏷️ POS & NER Using Transformer Models

> Part-of-speech tagging and named entity recognition powered by **Hugging Face transformer models** — more accurate than rule-based approaches, in just a few lines of code.

---

## 📂 Contents

| File | Task | Model |
|------|------|-------|
| `POS.ipynb` | Part-of-Speech tagging | `vblagoje/bert-english-uncased-finetuned-pos` |
| `NER.ipynb` | Named Entity Recognition | `dslim/bert-base-NER` |

---

## 🧠 How It Works

Both notebooks use the Hugging Face `pipeline("token-classification")` API:

```python
from transformers import pipeline

# POS tagging
pos = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", grouped_entities=True)
pos("I am meeting my friends for coffee this afternoon.")

# NER
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
ner("Apple is planning to open a new office in San Francisco next year.")
```

---

## 🆚 spaCy vs Transformer Models

| | spaCy | HF Transformer |
|---|-------|----------------|
| Speed | Fast | Slower (GPU helps) |
| Accuracy | Good | State-of-the-art |
| Setup | `spacy.load()` | `pipeline()` + model download |
| Best for | Production / real-time | Maximum accuracy |

---

## 🛠️ Requirements

```bash
pip install transformers torch
```

---

## ▶️ Run

```bash
jupyter notebook POS.ipynb
jupyter notebook NER.ipynb
```

---

## 🔁 Related

- [`../named entity recognition and extraction using spacy/`](../named%20entity%20recognition%20and%20extraction%20using%20spacy/) — NER with spaCy (rule + ML hybrid)
- [`../parsing/`](../parsing/) — POS & dependency parsing with spaCy

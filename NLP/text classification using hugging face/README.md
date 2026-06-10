# 🤗 Text Classification with Hugging Face

> Modern NLP in **3 lines of code** — pre-trained transformer pipelines for sentiment analysis, zero-shot classification and natural-language inference.

[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=flat)](https://huggingface.co/)

---

## 📂 Subfolders

| Folder | Pipeline | Use case |
|--------|----------|----------|
| 📂 [`sentiment analysis pipelines/`](./sentiment%20analysis%20pipelines/) | `sentiment-analysis` | Positive / negative classification |
| 📂 [`one shot classifier/`](./one%20shot%20classifier/) | `zero-shot-classification` | Classify into any labels — no fine-tuning needed |
| 📂 [`QNLI/`](./QNLI/) | QNLI model | Question-answer entailment |
| 📂 [`grammer checker using CoLA/`](./grammer%20checker%20using%20CoLA/) | `text-classification` | Grammatical acceptability checking |
| 📂 [`question similarity/`](./question%20similarity/) | `text-classification` | Detect duplicate / paraphrase questions (QQP) |

---

## ⚡ The Hugging Face `pipeline` API

Hugging Face's `pipeline()` abstracts away tokenization, model loading and post-processing:

```python
from transformers import pipeline

# Sentiment
classifier = pipeline("sentiment-analysis")
classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.999}]

# Zero-shot
classifier = pipeline("zero-shot-classification")
classifier(
    "I want to learn machine learning",
    candidate_labels=["education", "sports", "politics"]
)
# {'labels': ['education', 'politics', 'sports'], 'scores': [0.98, 0.01, 0.01]}
```

---

## 🛠️ Requirements

```bash
pip install transformers torch
```

> First run downloads the model (~250–500 MB). It's cached afterwards.

---

## ▶️ Run

```bash
jupyter notebook "sentiment analysis pipelines/p1.ipynb"
jupyter notebook "one shot classifier/p1.ipynb"
jupyter notebook "QNLI/p1.ipynb"
```

---

## 💡 Why Hugging Face?

- 🚀 **State-of-the-art** out of the box (BERT, RoBERTa, DistilBERT…)
- 🧪 **No fine-tuning required** for many tasks
- 🤝 **Huge community model hub** — pick a pretrained model for almost any language or domain

---

## 🧭 When to Use Which

| Task | Pipeline |
|------|---------|
| "Is this review positive or negative?" | `sentiment-analysis` |
| "Does this text fit one of these custom categories?" | `zero-shot-classification` |
| "Does this sentence answer this question?" | QNLI |
| "Translate / summarize / generate" | `translation`, `summarization`, `text-generation` |

# ❓ Question Answering

> Given a **context** paragraph and a **question**, extract or generate the answer — using pre-trained Hugging Face models.

---

## 📂 Contents

| File | QA Type | Model |
|------|---------|-------|
| `extractive QA.ipynb` | **Extractive** — highlights a span from the context | `distilbert-base-cased-distilled-squad` |
| `abstractive QA.ipynb` | **Abstractive** — generates a new sentence as the answer | `fangyuan/hotpotqa_abstractive` |

---

## 🆚 Extractive vs Abstractive

| | Extractive | Abstractive |
|---|-----------|-------------|
| Answer source | Copy-pasted from context | Generated from scratch |
| Pipeline | `question-answering` | `text2text-generation` |
| Pros | Faithful to source text | More natural, flexible |
| Cons | Can only answer if span exists | May hallucinate |

---

## ⚡ Quick Demo

```python
from transformers import pipeline

# Extractive
qa = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
qa(question="What is the display size?", context="This smartphone features a 6.5-inch OLED display.")
# {'answer': '6.5-inch', 'score': 0.87}

# Abstractive
gen = pipeline("text2text-generation", model="fangyuan/hotpotqa_abstractive")
gen("question: What is the display size? context: This smartphone features a 6.5-inch OLED display.")
# [{'generated_text': "The smartphone's display is a 6.5-inch OLED display."}]
```

---

## 🛠️ Requirements

```bash
pip install transformers torch
```

---

## ▶️ Run

```bash
jupyter notebook "extractive QA.ipynb"
jupyter notebook "abstractive QA.ipynb"
```

---

## 🔁 Related

- [`../text classification using hugging face/QNLI/`](../text%20classification%20using%20hugging%20face/QNLI/) — does a sentence answer a question? (entailment)
- [`../In class/RAG/`](../In%20class/RAG/) — combine retrieval with QA for open-domain answers

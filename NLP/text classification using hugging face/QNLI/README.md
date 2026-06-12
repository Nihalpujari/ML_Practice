# ❓ QNLI — Question Natural Language Inference

> Given a **question** and a **sentence**, decide whether the sentence **answers** (entails) the question.

QNLI is part of the [GLUE benchmark](https://gluebenchmark.com/) and a great demo of how transformer models can reason about question-answer relationships.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Load a QNLI-fine-tuned model and run inference on Q/A pairs |

---

## 🧠 The Task

| Premise (sentence) | Hypothesis (question) | Label |
|--------------------|------------------------|-------|
| *"Einstein developed the theory of relativity."* | *"Who developed the theory of relativity?"* | ✅ Entailment |
| *"Einstein loved playing the violin."* | *"Who developed the theory of relativity?"* | ❌ Not entailment |

Models fine-tuned on QNLI (e.g. RoBERTa, BERT) output a 2-way classification: **entailment** vs **not entailment**.

---

## ⚡ Demo

```python
from transformers import pipeline

# Use a QNLI-fine-tuned checkpoint
classifier = pipeline(
    "text-classification",
    model="cross-encoder/qnli-electra-base",
    tokenizer="cross-encoder/qnli-electra-base"
)

question = "Where was Einstein born?"
sentence = "Einstein was born in Ulm, Germany in 1879."

classifier(f"{question} </s> {sentence}")
```

---

## 🛠️ Requirements

```bash
pip install transformers torch
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 💡 Use Cases

- **Open-domain QA pipelines** — verify candidate passages actually answer the user's question
- **FAQ matching** — for each incoming question, score every FAQ entry
- **Reranking** — combine with a fast retrieval step (BM25 / dense vectors) → rerank top-k with QNLI

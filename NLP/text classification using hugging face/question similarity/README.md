# 🔄 Question Similarity (QQP)

> Are two questions asking the **same thing**? A BERT model fine-tuned on the **Quora Question Pairs** (QQP) dataset detects duplicate questions.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Classify whether two questions are paraphrases of each other |

---

## ⚡ Quick Demo

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="textattack/bert-base-uncased-QQP")

classifier({"text": "What's the process to change my password?",
            "text_pair": "How do I reset my account password?"})
# [{'label': 'LABEL_1', 'score': 0.98}]
# LABEL_1 = duplicate, LABEL_0 = not duplicate
```

The notebook also compares results with the DistilBERT variant (`textattack/distilbert-base-uncased-QQP`).

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

- **FAQ deduplication** — merge similar questions in a knowledge base
- **Search ranking** — detect when a new query matches an existing answer
- **Customer support** — route duplicate tickets to the same thread

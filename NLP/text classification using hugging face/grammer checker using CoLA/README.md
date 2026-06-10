# 📝 Grammar Checker Using CoLA

> Is this sentence grammatically acceptable? A BERT model fine-tuned on the **CoLA** (Corpus of Linguistic Acceptability) benchmark answers that question.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Classify sentences as grammatically acceptable or not |

---

## ⚡ Quick Demo

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")

classifier("Although she was knowing the answer, she didn't raised her hand.")
# [{'label': 'LABEL_1', 'score': 0.79}]
# LABEL_1 = acceptable, LABEL_0 = unacceptable
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

- Writing assistants and grammar checkers
- Automated essay scoring
- Filtering noisy text data before downstream NLP

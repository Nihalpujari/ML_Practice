# 😊😐😡 Sentiment Analysis Pipeline

> Classify text as **positive** or **negative** using a pre-trained Hugging Face model — in two lines of code.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Use `pipeline("sentiment-analysis")` on sample sentences |

---

## ⚡ Quick Demo

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

classifier("I love how fast this laptop is!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

classifier("The battery dies in 2 hours. Total waste of money.")
# [{'label': 'NEGATIVE', 'score': 0.9994}]
```

---

## 🧠 What's Happening Under the Hood

Default model: **`distilbert-base-uncased-finetuned-sst-2-english`**
1. Tokenize input with the DistilBERT tokenizer
2. Forward pass through the fine-tuned model
3. Apply softmax over `[NEGATIVE, POSITIVE]`
4. Return the top label + probability

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

## 💡 Tips

- Default model is **English only** — for other languages try `"nlptown/bert-base-multilingual-uncased-sentiment"`.
- The model gives **two labels** (POSITIVE / NEGATIVE) — if you want a **neutral** class, use a 3-class model.
- For **batch processing**, pass a list: `classifier(["text1", "text2", ...])`.

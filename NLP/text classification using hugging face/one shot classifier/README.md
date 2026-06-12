# 🎯 Zero-Shot Classification

> Classify any text into **any labels you make up** — no fine-tuning required.
> The trick: a Natural Language Inference (NLI) model that scores how well each label "follows from" the text.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Use `pipeline("zero-shot-classification")` to classify text into custom categories |

---

## ⚡ Quick Demo

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "I want to learn deep learning for natural language processing",
    candidate_labels=["technology", "sports", "politics", "food"]
)

# {
#   'labels': ['technology', 'politics', 'sports', 'food'],
#   'scores': [0.95, 0.02, 0.02, 0.01]
# }
```

---

## 🧠 How Does It Work?

The model treats classification as **textual entailment**:
> *"This text is about **{label}**"* — does the input text entail that statement?

For each candidate label, the model computes the probability that the hypothesis is entailed by the premise (your text). Highest probability wins.

Default model: **`facebook/bart-large-mnli`** (fine-tuned on MNLI).

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

- **Use `multi_label=True`** to allow multiple labels per text (not mutually exclusive).
- **Customize the hypothesis template** with `hypothesis_template`, e.g. `"This review is about a {} product"`.
- Zero-shot is a **convenience** — for production systems with stable labels, a small fine-tuned model is usually faster.

---

## 🎯 Great For

- Quick prototyping (no labeled training data)
- Routing support tickets by topic
- Tagging short user-generated content
- Bootstrapping a labeled dataset (then fine-tune later)

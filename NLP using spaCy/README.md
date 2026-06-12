# 🔤 NLP using spaCy

> A quick-start introduction to **spaCy** — downloading models, loading pipelines, and tokenizing text.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `download and load models.ipynb` | Install and load a spaCy language model (`en_core_web_sm`) |
| `tokenization.ipynb` | Tokenize text using spaCy's NLP pipeline |

---

## 🎯 What You'll Learn

1. Install spaCy and download a pre-trained English model
2. Load the model with `spacy.load("en_core_web_sm")`
3. Process text through the spaCy pipeline to get `Doc` objects
4. Access individual tokens — text, POS tags, lemmas, stop-word flags

---

## 🛠️ Requirements

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## ▶️ Run

```bash
jupyter notebook "download and load models.ipynb"
jupyter notebook tokenization.ipynb
```

---

## 💡 Tip

For a deeper dive into spaCy's NLP features (NER, dependency parsing, custom pipelines), see [`NLP/named entity recognition and extraction using spacy/`](../NLP/named%20entity%20recognition%20and%20extraction%20using%20spacy/) and [`NLP/parsing/`](../NLP/parsing/).

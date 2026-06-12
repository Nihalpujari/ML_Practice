# 🔮 Sequence Generation Tasks

> Three common **text generation** tasks powered by Hugging Face pipelines — search completion, summarization and translation.

---

## 📂 Contents

| File | Task | Model |
|------|------|-------|
| `search completion system.ipynb` | Autocomplete search queries | `distilgpt2` |
| `text summarization.ipynb` | Condense articles into key points | `cnicu/t5-small-booksum` |
| `translation.ipynb` | English → French translation | `Helsinki-NLP/opus-mt-en-fr` |

---

## 🧠 The Three Tasks

### 1. Search Completion
Feed a partial query to a **causal language model** and let it predict the next tokens:
```python
pipeline("text-generation", model="distilgpt2")("Best books to read for", num_return_sequences=5)
```

### 2. Text Summarization
Compress a long article into a short summary using a **seq2seq** model:
```python
pipeline("summarization", model="cnicu/t5-small-booksum")(article)
```

### 3. Translation
Translate text between languages using a **MarianMT** model:
```python
pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")(text)
```

---

## 🛠️ Requirements

```bash
pip install transformers torch
```

---

## ▶️ Run

```bash
jupyter notebook "search completion system.ipynb"
jupyter notebook "text summarization.ipynb"
jupyter notebook translation.ipynb
```

---

## 💡 Tips

- **Temperature** and **top-k/top-p** sampling control creativity vs determinism in text generation.
- For production summarization, `facebook/bart-large-cnn` is a strong default.
- Translation models are language-pair specific — browse `Helsinki-NLP/opus-mt-*` on Hugging Face for other pairs.

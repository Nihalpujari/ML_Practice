# 🔠 Tokenization

> Tokenization is the **very first step** in any NLP pipeline — slicing raw text into smaller, meaningful units (tokens).

---

## 📂 Contents

| File | Topic |
|------|-------|
| `p1.py` | **Word** tokenization with `nltk.word_tokenize()` |
| `p2.py` | **Sentence** tokenization with `nltk.sent_tokenize()` |
| `p2.ipynb` | Notebook version exploring both |

---

## 📝 Sample Input

```
The stock market saw a significant dip today. Experts believe the downturn may continue.
However, many investors are optimistic about future growth.
```

## 🎯 Word Tokenization — `p1.py`
Splits the paragraph into a list of **words and punctuation**:

```python
import nltk
tokens = nltk.word_tokenize(text)
# ['The', 'stock', 'market', 'saw', 'a', 'significant', 'dip', 'today', '.', ...]
```

## 🎯 Sentence Tokenization — `p2.py`
Splits the paragraph into a list of **sentences**:

```python
import nltk
sentences = nltk.sent_tokenize(text)
# ['The stock market saw a significant dip today.',
#  'Experts believe the downturn may continue.', ...]
```

---

## 🛠️ Requirements

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```

---

## ▶️ Run

```bash
python p1.py
python p2.py
jupyter notebook p2.ipynb
```

---

## 💡 Note

⚠️ **Heads-up:** `p1.py` is labeled "sentence tokenization" in a comment but actually calls `word_tokenize()` — that's the intended exercise. `p2.py` is the proper sentence tokenizer.

---

## 🆚 Tokenizer Options

| Tool | Strength |
|------|---------|
| `str.split()` | Fast but naive — splits on whitespace only |
| `nltk.word_tokenize` | Handles punctuation, contractions |
| `nltk.sent_tokenize` | Sentence-level segmentation |
| `spacy` | Rule + ML hybrid, language-aware |
| `transformers` (BPE/WordPiece) | Sub-word for modern LLMs |

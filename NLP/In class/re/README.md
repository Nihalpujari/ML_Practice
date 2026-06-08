# 🔤 Regular Expressions for NLP

> The Swiss Army knife of text processing — `re` extracts patterns (emails, URLs, dates, numbers, hashtags…) before any "real" NLP starts.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `re.ipynb` | Pure Python `re` patterns — search, match, findall, sub |
| `re_nltk_punkt.ipynb` | Combine regex with NLTK's punkt tokenizer |

---

## 🎯 What You'll Learn

### `re.ipynb`
- `re.search()` vs `re.match()` vs `re.findall()`
- Character classes — `\d`, `\w`, `\s`
- Quantifiers — `*`, `+`, `?`, `{n,m}`
- Anchors — `^`, `$`, `\b`
- Capture groups `(…)` and replacements with `re.sub()`
- Common patterns: emails, URLs, phone numbers, hashtags

### `re_nltk_punkt.ipynb`
- Use `nltk.tokenize.PunktSentenceTokenizer` for sentence segmentation
- Combine regex pre-cleaning with NLTK tokenization
- Train Punkt on a custom corpus (advanced)

---

## ⚡ Cheat Sheet

| Pattern | Matches |
|---------|---------|
| `\d+` | One or more digits |
| `\w+` | One or more word characters |
| `[A-Za-z]+` | Letters only |
| `\b\w+\b` | Whole words |
| `https?://\S+` | URLs |
| `\b[\w.+-]+@[\w-]+\.[\w.-]+\b` | Emails |
| `#\w+` | Hashtags |
| `@\w+` | Mentions |

---

## 🛠️ Requirements

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```

---

## ▶️ Run

```bash
jupyter notebook re.ipynb
jupyter notebook re_nltk_punkt.ipynb
```

---

## 💡 Tip

Test regex patterns on [regex101.com](https://regex101.com/) — it shows live matches and explains each part of your pattern.

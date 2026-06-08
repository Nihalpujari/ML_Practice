# 🧭 Parsing — POS & Dependency Parsing

> **Parsing** uncovers the *grammatical structure* of a sentence — which words are nouns, verbs, subjects, objects, etc.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `p1.ipynb` | Part-of-speech tagging & dependency parsing with spaCy |

---

## 🧠 What Parsing Gives You

### Part-of-Speech (POS) Tagging
Labels each token with its grammatical role:

| Word | POS | Meaning |
|------|-----|---------|
| `The` | DET | Determiner |
| `quick` | ADJ | Adjective |
| `fox` | NOUN | Noun |
| `jumps` | VERB | Verb |

### Dependency Parsing
Shows **how words relate to each other** as a tree:

```
jumps
├── nsubj → fox
│   └── amod → quick
│   └── det → The
└── prep → over
```

This unlocks features like:
- Finding the **subject** of a sentence (who is doing the action?)
- Extracting **noun phrases**
- Building **rule-based information extractors**

---

## 🛠️ Requirements

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 💡 Key spaCy Snippets

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# POS tags
for token in doc:
    print(token.text, token.pos_, token.tag_)

# Dependency tree
for token in doc:
    print(token.text, "→", token.dep_, "→", token.head.text)

# Visualize
from spacy import displacy
displacy.render(doc, style="dep")
```

---

## 🔁 Related

- [`../named entity recognition and extraction/`](../named%20entity%20recognition%20and%20extraction/) — extract people, places, organizations

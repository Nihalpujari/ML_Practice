# 🏷️ Named Entity Recognition (NER)

> **NER** identifies and classifies *named things* in text — people, places, organizations, dates, money, etc.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `p1.ipynb` | NER with spaCy — extract entities from sample text |

---

## 🧠 Example

Input:
> *"Tim Cook, the CEO of Apple, said in California on January 5th that the company will invest $430 billion in the US over five years."*

Output:
| Entity | Type | Description |
|--------|------|-------------|
| Tim Cook | PERSON | People |
| Apple | ORG | Organization |
| California | GPE | Geo-political entity |
| January 5th | DATE | Date |
| $430 billion | MONEY | Monetary value |
| five years | DATE | Duration |

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

## 💡 Key spaCy NER Snippets

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.label_)
# Apple ORG
# U.K. GPE
# $1 billion MONEY

# Visualize
from spacy import displacy
displacy.render(doc, style="ent")
```

---

## 🎯 Use Cases

- **Resume parsing** — pull names, skills, organizations from CVs
- **News analytics** — track which people / companies are mentioned together
- **Search & filter** — let users search documents by entity
- **Knowledge graphs** — populate nodes & edges from raw text

---

## 🔁 Related

- [`../parsing/`](../parsing/) — syntactic structure for richer extraction
- [`../jobs_applicants_project/`](../jobs_applicants_project/) — applied to applicant data

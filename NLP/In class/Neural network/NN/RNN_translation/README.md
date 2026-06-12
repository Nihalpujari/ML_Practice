# 🌐 RNN Translation Experiments

> Two stepping-stone notebooks on building **sequence-to-sequence translation** with RNNs.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `RNN_TranslateFirstWord.ipynb` | Toy task — predict only the *first* word of the target language |
| `RNNTranslationTimedistributed.ipynb` | Full-sequence translation with `TimeDistributed` outputs |

---

## 🧠 The Two Modes

### 1. First-Word Prediction (`RNN_TranslateFirstWord.ipynb`)
- Encode the source sentence with an RNN
- Read out the **last hidden state**
- Predict only **one word** (the first word of the translation)
- 👶 Great for **understanding the mechanics** without juggling decoder logic

### 2. Time-Distributed Output (`RNNTranslationTimedistributed.ipynb`)
- Encode the source with an RNN
- Apply a `TimeDistributed(Dense)` layer to produce **a word at each timestep**
- A pragmatic stepping stone toward full **encoder–decoder** seq2seq

---

## 🧱 Typical Pipeline

```
1. Load parallel corpus  (English ↔ target language pairs)
2. Tokenize both sides
3. Pad sequences to a fixed length
4. Build vocabulary indexes
5. Build the model — Embedding → LSTM → [TimeDistributed] Dense → softmax
6. Train with categorical cross-entropy
7. Decode predictions back to text
```

---

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook RNN_TranslateFirstWord.ipynb
jupyter notebook RNNTranslationTimedistributed.ipynb
```

---

## 💡 Tips & Limitations

- Real translation needs **encoder–decoder + attention** (or a Transformer) — these notebooks are *intentionally* simpler.
- **Vocabulary size matters** — a 5k–20k word vocabulary is a good starting point for toy data.
- For production-quality translation today: use a pre-trained model from Hugging Face (`Helsinki-NLP/opus-mt-*`).

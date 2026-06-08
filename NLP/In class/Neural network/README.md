# 🧠 Neural Networks for NLP

> Move beyond classical ML — neural networks for text classification, sequence learning, and translation.

---

## 📂 Folder Structure

```
Neural network/
│
├── p1.ipynb            Top-level NN intro / experiment
│
└── 📂 NN/
    └── 📂 RNN_translation/
        ├── RNN_TranslateFirstWord.ipynb
        └── RNNTranslationTimedistributed.ipynb
```

---

## 📓 Notebook Highlights

### `p1.ipynb` — Neural Network Intro
- Build a simple feed-forward classifier on text features (BoW / TF-IDF vectors)
- Train with Keras (`Dense` layers, `relu`, `softmax`)
- Compare against classical ML baselines (Logistic Regression / SVM)

### 📂 `NN/RNN_translation/`
Two RNN-based translation experiments:

#### `RNN_TranslateFirstWord.ipynb`
- A **toy** translation: predict only the *first* word of the target language
- Great for understanding sequence-to-sequence mechanics step by step

#### `RNNTranslationTimedistributed.ipynb`
- Full sequence translation using **`TimeDistributed`** dense output
- Produces a word for every input timestep
- Bridges the gap to a proper **encoder–decoder** architecture

---

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
jupyter notebook NN/RNN_translation/RNN_TranslateFirstWord.ipynb
jupyter notebook NN/RNN_translation/RNNTranslationTimedistributed.ipynb
```

---

## 💡 Tips

- Always **pad/truncate sequences** to a uniform length before feeding an RNN (`pad_sequences`).
- **Mask padding** so the model doesn't learn from those positions.
- For real translation, look at **encoder–decoder + attention** (the precursor to Transformers).

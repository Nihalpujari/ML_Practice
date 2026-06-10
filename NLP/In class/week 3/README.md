# 🧪 Week 3 — MLP vs RNN vs LSTM vs GRU

> Compare four neural network architectures on the **diabetes classification** task using PyTorch. Spoiler: MLP wins on tabular data — but the exercise teaches you *why*.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `NLP_Lab_Week3_.ipynb` | Full lab — build MLP, RNN, LSTM, GRU from scratch in PyTorch and compare |
| `NLP_Lab_Week3_exercises_.ipynb` | Exercise version with fill-in-the-blank cells |
| `bodyPerformance.csv` | Alternative dataset for extra practice |

---

## 🧠 Architectures Compared

| Model | Type | Gates | Long-range memory |
|-------|------|-------|-------------------|
| **MLP** | Feed-forward | — | — |
| **RNN** | Recurrent | 0 | No (vanishing gradients) |
| **LSTM** | Recurrent | 3 (forget, input, output) | Yes |
| **GRU** | Recurrent | 2 (reset, update) | Yes |

Each model is built in PyTorch with `nn.Module`, trained with `BCEWithLogitsLoss`, and evaluated with accuracy, precision, recall, F1, and ROC-AUC.

---

## 🎯 Key Takeaway

The diabetes dataset is **tabular** (no temporal ordering), so the MLP typically outperforms the recurrent models. RNNs, LSTMs, and GRUs shine on genuinely **sequential** data (text, time series, speech).

---

## 🛠️ Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook NLP_Lab_Week3_.ipynb
```

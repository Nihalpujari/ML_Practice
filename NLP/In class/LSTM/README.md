# 🧠 LSTM — Long Short-Term Memory

> RNNs forget. **LSTMs remember.**
> An LSTM cell adds gates that decide what to keep, what to forget, and what to output — solving the vanishing-gradient problem of vanilla RNNs.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Build & train an LSTM on a sequence task |

---

## 🧠 The Three Gates of an LSTM

| Gate | What it does |
|------|-------------|
| **Forget gate** `f_t` | What to drop from cell memory |
| **Input gate** `i_t` | What new info to write into memory |
| **Output gate** `o_t` | What to expose to the next layer |

Internally, the **cell state `c_t`** flows through with minimal multiplicative interference — that's why gradients survive over long sequences.

---

## 🎯 What You'll Learn

1. Why vanilla RNNs struggle with long-range dependencies
2. The gate equations of an LSTM cell
3. Building an LSTM in Keras with `tf.keras.layers.LSTM`
4. Training on a sequence task (text classification, character generation, etc.)
5. Monitoring training & validation loss / accuracy

---

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook p1.ipynb
```

---

## 🆚 RNN vs LSTM vs GRU

| | RNN | LSTM | GRU |
|---|-----|------|-----|
| Gates | 0 | 3 | 2 |
| Long-range memory | ❌ | ✅ | ✅ |
| Parameters | Fewest | Most | In between |
| When to pick | Almost never | Default sequence model (pre-Transformer) | Faster LSTM alternative |

---

## 💡 Tip

LSTMs are still **excellent** for small datasets or low-latency applications. Don't reach for a Transformer when an LSTM will do.

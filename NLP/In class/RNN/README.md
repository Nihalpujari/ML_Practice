# 🔁 RNN — Recurrent Neural Networks

> An RNN processes a sequence **one token at a time**, carrying a hidden state forward.
> Perfect intuition-builder before you tackle LSTMs and Transformers.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `step_by_step_RNN.ipynb` | A step-by-step build of an RNN — see exactly what happens at each timestep |

---

## 🧠 The Idea

At every timestep `t`:

```
h_t = tanh(W_x · x_t + W_h · h_(t-1) + b)
y_t = W_y · h_t + b_y
```

- `x_t` — input at step t
- `h_t` — hidden state at step t (the "memory")
- `y_t` — output at step t

The same weights `W_x, W_h, W_y` are reused at every step — that's what makes it *recurrent*.

---

## 🎯 What You'll Learn

1. Why feed-forward networks can't handle sequences well
2. How an RNN unrolls over time
3. The **vanishing-gradient** problem (why LSTMs were invented)
4. Building a tiny RNN by hand in NumPy / Keras
5. Training it on a simple sequence task

---

## 🛠️ Requirements

```bash
pip install numpy tensorflow keras matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook step_by_step_RNN.ipynb
```

---

## 🔁 Up next

- [`../LSTM/`](../LSTM/) — RNN's better-behaved cousin
- [`../Transformer_keras/`](../Transformer_keras/) — the architecture that retired RNNs
- [`../Neural network/NN/RNN_translation/`](../Neural%20network/NN/RNN_translation/) — RNNs applied to translation

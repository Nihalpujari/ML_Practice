# 🤖 Transformer (Keras)

> The architecture that **retired RNNs** for most NLP tasks — built from scratch in Keras.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `p1.ipynb` | Build a (mini) Transformer encoder/decoder in TensorFlow / Keras |

---

## 🧠 Core Ideas

### Self-Attention
For each token, compute weighted relationships to every other token:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

- `Q` = Queries
- `K` = Keys
- `V` = Values

### Multi-Head Attention
Run several attention heads in parallel — each head can focus on different relationships.

### Positional Encoding
Since attention is order-agnostic, we **inject position information** via sinusoidal embeddings (or learned positional embeddings).

### Block Architecture
```
[Input + Positional Encoding]
       ↓
[Multi-Head Self-Attention]
       ↓ + residual + LayerNorm
[Feed-Forward Network]
       ↓ + residual + LayerNorm
[Repeat N times]
```

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

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original Vaswani et al. paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [Keras NLP Transformer tutorial](https://keras.io/examples/nlp/text_classification_with_transformer/)

---

## 💡 Tip

Once you understand a tiny Transformer block, you can fluently read **GPT, BERT, T5, ViT** papers — they're all stacks of these same blocks with task-specific heads.

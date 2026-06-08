# 🌸 K-Means on the Iris Dataset

The "Hello, World" of clustering — the Iris dataset has 3 species, so we expect `k = 3` to recover them naturally from petal & sepal measurements.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `iris_data.ipynb` | Load → scale → fit K-Means → visualize clusters |

---

## 📊 Dataset

The classic [Fisher Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) — 150 flowers, 3 species, 4 features each:

| Feature | Unit |
|---------|------|
| Sepal length | cm |
| Sepal width | cm |
| Petal length | cm |
| Petal width | cm |

Use `sklearn.datasets.load_iris()` to load it directly.

---

## 🎯 What You'll Learn

1. Load the dataset with `load_iris()`
2. Scale features with `StandardScaler`
3. Apply K-Means with `n_clusters=3`
4. Compare cluster labels with true species (which we **pretend we don't have** while clustering)
5. Plot clusters in 2D using two of the four features

---

## ▶️ Run

```bash
jupyter notebook iris_data.ipynb
```

---

## 💡 Why This Dataset?

- **Small** — fast to iterate on
- **Well-separated classes** — K-Means tends to "succeed" visibly
- **Famous benchmark** — easy to compare your results against textbooks

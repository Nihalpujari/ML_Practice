# 🚀 Advanced Supervised Learning

> A notebook focused on the next layer up from "plain sklearn fit/predict":
> pipelines, multiple-model comparison, and richer evaluation.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `code.ipynb` | Advanced supervised-learning techniques and model comparison |

---

## 🎯 What This Covers

Typical themes for an "advanced" supervised notebook:

- **`sklearn.pipeline.Pipeline`** — chain preprocessing + model so leakage can't sneak in
- **Multiple algorithms side-by-side** — Logistic Regression vs Random Forest vs SVM vs Gradient Boosting
- **Cross-validation** with `cross_val_score` for fair comparisons
- **Evaluating beyond accuracy** — precision, recall, F1, ROC-AUC, PR curves
- **Saving / loading models** with `joblib`

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 💡 Tip

When comparing models, **always evaluate them on the same cross-validation folds** (use a fixed `cv` object). Otherwise a "win" might just be variance from different splits.

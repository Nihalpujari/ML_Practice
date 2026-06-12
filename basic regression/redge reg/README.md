# 🧷 Ridge Regression (L2)

> Linear regression's well-behaved cousin — shrinks coefficients smoothly without ever forcing them to zero.

Ridge regression adds an **L2 penalty** on the squared sum of coefficients, helping prevent overfitting when features are correlated or numerous.

---

## 📂 Files

| File | Purpose |
|------|---------|
| `code redge.ipynb` | Fit Ridge regression, compare with plain Linear Regression |
| `student_exam_dataset.csv` | Tiny dataset — `Hours_Studied, Attendance_Percentage, Pass` |

---

## 🧠 The Idea

```
Loss = MSE + α · Σ(wᵢ²)
```

| `α` | Effect |
|-----|--------|
| `0` | Identical to ordinary linear regression |
| Small | Mild coefficient shrinkage |
| Large | Strong shrinkage — coefficients approach (but never reach) zero |

Unlike Lasso, Ridge **never zeros features** — it just makes them smaller. Use it when you believe **all features carry some signal**.

---

## 🎯 What You'll Learn

1. Use `sklearn.linear_model.Ridge(alpha=...)`
2. See how coefficients change as `α` grows
3. Compare R² on train vs test to detect overfitting
4. Understand why scaling matters before regularization

---

## ▶️ Run

```bash
jupyter notebook "code redge.ipynb"
```

---

## 💡 Ridge vs Lasso — Quick Decision

| Scenario | Pick |
|----------|------|
| Few features, all believed relevant | Ridge |
| Many features, expect some to be useless | Lasso |
| You want both shrinkage *and* selection | ElasticNet (Ridge + Lasso) |

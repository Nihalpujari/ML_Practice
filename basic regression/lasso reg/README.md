# 🪢 Lasso Regression (L1)

> *"When in doubt, Lasso will zero it out."*

Lasso (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) is linear regression with an **L1 penalty**. Its superpower: it drives unimportant coefficients to *exactly* zero, performing **automatic feature selection**.

---

## 📂 Files

| File | Purpose |
|------|---------|
| `code.ipynb` | Fit Lasso, inspect coefficients, plot results |
| `diabetes_clean.csv` | Pima Indians Diabetes dataset |

---

## 🧠 The Idea

Standard linear regression minimizes only the **mean squared error**:

```
Loss = MSE
```

Lasso adds an L1 penalty on the coefficient magnitudes:

```
Loss = MSE + α · Σ|wᵢ|
```

- `α = 0` → ordinary linear regression
- `α → ∞` → all coefficients shrink to zero
- For moderate `α`, **weak features get exactly zero weight** → built-in feature selection

---

## 🎯 What You'll Learn

1. Fit `sklearn.linear_model.Lasso(alpha=...)`
2. Read `model.coef_` and identify which features survived
3. Visualize coefficient values as a bar chart
4. Compare against ordinary `LinearRegression` to see the shrinkage effect

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 💡 Tips

- **Scale your features** before fitting — Lasso's penalty is scale-sensitive.
- Use **`LassoCV`** to automatically pick the best `α` via cross-validation.
- A Lasso coefficient plot is a fantastic way to *explain* which features your model relies on.

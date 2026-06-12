# 🎯 Logistic Regression

> Don't let the name fool you — **logistic regression is a classifier**, not a regressor.
> It predicts the probability that an input belongs to a class via the sigmoid function.

---

## 📂 Files

| File | Purpose |
|------|---------|
| `logistic reg.ipynb` | Train logistic regression on diabetes data |
| `dataset_with_names.csv` | Diabetes dataset with an extra `Person_Name` column |

---

## 📊 Dataset

Same 8 clinical features as the Pima Indians dataset, plus:

| Column | Meaning |
|--------|---------|
| `Person_Name` | Identifier (dropped before fitting) |
| `pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age` | Numeric features |
| `diabetes` | Binary target: 1 = diabetic, 0 = not |

---

## 🧠 The Idea

For a linear combination of features `z = w·x + b`, logistic regression squashes it through the **sigmoid**:

```
σ(z) = 1 / (1 + e^(-z))
```

Output is a probability in `[0, 1]` — anything ≥ 0.5 is class 1 (by default).

---

## 🎯 What You'll Learn

1. Why we can't just use linear regression for 0/1 targets
2. Fitting `sklearn.linear_model.LogisticRegression`
3. Predicting class labels with `.predict()` and probabilities with `.predict_proba()`
4. Evaluating with:
   - **Accuracy**
   - **Confusion matrix**
   - **Precision / Recall / F1**
5. Interpreting coefficients as **log-odds**

---

## ▶️ Run

```bash
jupyter notebook "logistic reg.ipynb"
```

---

## 💡 Tips

- **Drop the `Person_Name` column** before fitting — it's an ID, not a feature.
- **Scale features** (`StandardScaler`) for better convergence.
- Adjust the **classification threshold** (not always 0.5!) if your classes are imbalanced or recall matters more than precision.

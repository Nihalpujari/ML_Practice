# 🩺 Linear Regression on the Diabetes Dataset

A starter notebook using the **Pima Indians Diabetes** dataset to learn the mechanics of `sklearn.linear_model.LinearRegression`.

---

## 📂 Files

| File | Purpose |
|------|---------|
| `code.ipynb` | Main notebook — load, fit, evaluate |
| `diabetes_clean.csv` | Cleaned dataset (768 rows, 9 columns) |

---

## 📊 Dataset

| Column | Meaning |
|--------|---------|
| `pregnancies` | Number of pregnancies |
| `glucose` | Plasma glucose concentration |
| `diastolic` | Diastolic blood pressure (mm Hg) |
| `triceps` | Triceps skin fold thickness (mm) |
| `insulin` | 2-hour serum insulin |
| `bmi` | Body Mass Index |
| `dpf` | Diabetes Pedigree Function |
| `age` | Age in years |
| `diabetes` | Outcome (0 = no, 1 = yes) |

---

## 🎯 What You'll Learn

1. **Loading data** with pandas and exploring with `df.describe()`
2. **Splitting features (X) and target (y)**
3. **Train/test split** with `train_test_split`
4. **Fitting** `LinearRegression()` and reading `coef_` / `intercept_`
5. **Evaluating** with R², MAE and RMSE

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 💡 Note

The `diabetes` column is binary (0/1), so technically **logistic regression** would be the more correct model. Treating it as a regression target is fine for *practising* the linear-regression API — just don't read too much into the predicted values.

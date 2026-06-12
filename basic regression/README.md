# 📈 Basic Regression

> Regression is the bread-and-butter of ML — fit a curve, predict a number, understand the relationship.
> This folder walks through **linear, regularized and logistic** regression on small, focused datasets.

---

## 📂 Folder Structure

```
basic regression/
│
├── 📂 diabetes/        Simple & multiple linear regression on diabetes data
├── 📂 lasso reg/       L1 regularization (feature selection)
├── 📂 redge reg/       L2 regularization (coefficient shrinkage)
├── 📂 logistic reg/    Binary classification with logistic regression
└── 📄 visual CV Lreg.py  Visualize 6-fold CV scores with mean & 95% CI
```

---

## 📚 Topics by Subfolder

### 📂 `diabetes/` — Linear Regression
Predict the `diabetes` outcome (or other numeric targets) from clinical features such as glucose, BMI, blood pressure, etc.

- **Dataset:** `diabetes_clean.csv` — the classic Pima Indians Diabetes dataset
- **Columns:** `pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age, diabetes`
- **Techniques:** Simple linear regression → multiple linear regression
- **Metrics:** R², MAE, MSE, RMSE

---

### 📂 `redge reg/` — Ridge Regression (L2)
Adds a penalty proportional to the **squared** sum of coefficients. Shrinks weights toward zero — useful when features are correlated.

- **Dataset:** `student_exam_dataset.csv` (`Hours_Studied, Attendance_Percentage, Pass`)
- **Notebook:** `code redge.ipynb`
- **Key idea:**
  ```
  Loss = MSE + α * Σ(coef²)
  ```
  Higher `α` → smaller coefficients → more bias, less variance.

---

### 📂 `lasso reg/` — Lasso Regression (L1)
Adds a penalty proportional to the **absolute** sum of coefficients. Drives weak coefficients to **exactly zero**, performing **feature selection**.

- **Dataset:** `diabetes_clean.csv`
- **Notebook:** `code.ipynb`
- **Key idea:**
  ```
  Loss = MSE + α * Σ|coef|
  ```
- **Bonus:** Plot coefficient magnitudes to see which features Lasso zeroed out.

---

### 📂 `logistic reg/` — Logistic Regression
Despite the name, this is a **classification** algorithm. Outputs a probability via the sigmoid function.

- **Dataset:** `dataset_with_names.csv` — same diabetes features + a `Person_Name` column
- **Notebook:** `logistic reg.ipynb`
- **Target:** Binary `diabetes` (0 / 1)
- **Evaluation:** Accuracy, confusion matrix, precision / recall

---

### 📄 `visual CV Lreg.py` — Cross-Validation Visualization

A standalone matplotlib script that takes a list of 6-fold CV scores and plots:
- 🔵 individual fold scores
- 🟢 mean R²
- 🟢 shaded **95% confidence interval** (using `1.96 × SE`)

```bash
python "visual CV Lreg.py"
```

Use this to sanity-check whether your model's CV performance is stable across folds — a high variance band is a red flag.

---

## 🛠️ Requirements

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

---

## 🧠 Cheat Sheet

| Algorithm | When to use | Penalty | sklearn class |
|-----------|-------------|---------|--------------|
| Linear Regression | Baseline, interpretable | None | `LinearRegression` |
| Ridge | Many correlated features | L2 (squared) | `Ridge` |
| Lasso | Want automatic feature selection | L1 (absolute) | `Lasso` |
| Logistic Regression | Binary classification | None / L1 / L2 | `LogisticRegression` |

---

## 💡 Tips

- **Always scale features** before fitting Ridge / Lasso (their penalty is scale-sensitive).
- **Use cross-validation**, not just a single train/test split, to pick `α`.
- `LassoCV` and `RidgeCV` will pick the best `α` for you in one line.

---

🔁 **Next step:** explore [`../basic knn/`](../basic%20knn/) for your first non-parametric algorithm.

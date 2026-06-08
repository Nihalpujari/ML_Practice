# 🧹 Data Preprocessing

> The first, often-unglamorous, **most important** step in any ML pipeline.
> Bad data ⇒ bad model — no matter how fancy the algorithm.

This folder covers techniques for cleaning raw data and preparing it for downstream models.

---

## 📂 Contents

| File | Type | What it covers |
|------|------|----------------|
| `categorical features.ipynb` | 📓 Notebook | Encoding text/categorical columns into numbers — Label Encoding, One-Hot Encoding, `pd.get_dummies()` |
| `train.csv` | 📄 Dataset | Titanic training set — mixed numeric & categorical features (used as practice ground) |
| `test.csv` | 📄 Dataset | Titanic test set |
| `github_candidates_1.csv` | 📄 Dataset | GitHub user-profile data — public repos, stars, language, CI/CD usage |

---

## 🎯 Topics Covered

### 1. Categorical Encoding
When a column contains text labels (e.g. `male` / `female`, `Java` / `Python`), most ML models can't process it directly. This notebook compares:

- **Label Encoding** — maps each category to an integer (good for ordinal data)
- **One-Hot Encoding** — creates a binary column per category (good for nominal data)
- **`pd.get_dummies()`** — pandas' quick one-hot shortcut

### 2. Working with Mixed-Type Datasets
The Titanic CSVs include:
- Numeric features (`Age`, `Fare`)
- Binary/categorical features (`Sex`, `Embarked`)
- Free-text / high-cardinality columns (`Name`, `Ticket`, `Cabin`)
- **Missing values** in `Age`, `Cabin`, `Embarked`

### 3. The GitHub Candidates Dataset
A toy 10-row dataset modelling developer profiles:

| Column | Description |
|--------|-------------|
| `username` | GitHub handle |
| `public_repos` | Number of public repos |
| `total_stars` | Cumulative stars across all repos |
| `total_forks` | Cumulative forks |
| `commits_12m` | Commits in the last 12 months |
| `dominant_language` | Most-used language (categorical) |
| `has_cicd` | Binary — does the user use CI/CD? |

Useful for practising encoding `dominant_language` and scaling the numeric columns.

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn jupyter
```

---

## ▶️ Run

```bash
jupyter notebook "categorical features.ipynb"
```

---

## 💡 Key Takeaways

- **Always inspect first.** Run `df.info()` and `df.isna().sum()` before doing anything.
- **Don't blindly drop columns.** A column with lots of missing values may still carry signal (e.g. Titanic's `Cabin`).
- **One-hot encoding can explode dimensionality** on high-cardinality columns — consider grouping rare categories.
- **Fit encoders on train only**, then `transform` test — otherwise you leak information.

---

🔁 **Up next:** once your data is clean, head over to [`../basic regression/`](../basic%20regression/) to train your first models.

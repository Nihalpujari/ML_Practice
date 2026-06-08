# ⚙️ Hyperparameter Tuning

> A model has two kinds of "knobs":
> - **Parameters** — learned during training (e.g. weights of a linear regression)
> - **Hyperparameters** — set *before* training (e.g. `k` for KNN, `α` for Ridge)
>
> Picking the right hyperparameters is often what separates a *meh* model from a great one.

This folder shows how to **systematically search** for the best hyperparameters using scikit-learn's `GridSearchCV`.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `GridSearchCV.ipynb` | Exhaustive grid search over a parameter grid |
| `UsingHyperPArarmeterForKnn.ipynb` | Hands-on KNN tuning — find the best `k` (and more) |
| `diabetes_clean.csv` | Pima Indians Diabetes dataset (used by both notebooks) |

---

## 🧠 The Tuning Workflow

```
1. Define the model
2. Define the parameter grid          {param: [v1, v2, v3], ...}
3. Wrap in GridSearchCV(cv=k)
4. .fit(X, y)
5. Read .best_params_ and .best_score_
```

`GridSearchCV` automatically:
- Splits the data into `cv` folds
- Trains the model with **every combination** of parameters on each fold
- Returns the combo with the best average score

---

## 📓 Notebook Highlights

### `GridSearchCV.ipynb`
- Build a parameter grid (e.g. for KNN: `n_neighbors`, `weights`, `metric`)
- Run `GridSearchCV` with 5-fold CV
- Inspect `cv_results_` for full performance breakdown
- Pick the winner with `best_estimator_`

### `UsingHyperPArarmeterForKnn.ipynb`
- Focused KNN tuning on the diabetes dataset
- Loop through a range of `k` values **manually** and plot accuracy
- Then automate it with `GridSearchCV`
- Compare manual vs automated results

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook GridSearchCV.ipynb
jupyter notebook UsingHyperPArarmeterForKnn.ipynb
```

---

## 🆚 Grid Search vs Random Search

| | `GridSearchCV` | `RandomizedSearchCV` |
|---|----------------|---------------------|
| Strategy | Tries **every** combination | Samples `n_iter` random combinations |
| Speed | Slow when grid is large | Much faster |
| Best for | Small, focused grids | Wide / continuous parameter spaces |

For big search spaces, **start with Randomized** to zoom in, then **finish with Grid** in that smaller window.

---

## 💡 Tips

- **Always use cross-validation** during tuning — a single split lies to you.
- **Don't tune on the test set** — keep a held-out test set untouched until the very end.
- Use **`Pipeline`** so scaling + model fit happen *inside* each CV fold (no data leakage).
- More folds = more reliable scores but more compute. `cv=5` is a sensible default.

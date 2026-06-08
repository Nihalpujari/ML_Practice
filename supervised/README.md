# 🧠 Supervised Learning — Master Folder

> A **consolidated workspace** where every step of the supervised-learning journey lives in one place.
> Think of this folder as your "all-in-one playground" — preprocessing, KNN, regression, tuning, projects, all together.

---

## 📂 Folder Structure

```
supervised/
│
├── 📂 data preprocessing/      Encoding, scaling, working with Titanic & GitHub data
├── 📂 basic knn/               KNN classifier + decision boundary visualization
├── 📂 basic regression/        Linear, Ridge, Lasso, Logistic regression
├── 📂 Hyperparameter tuning/   GridSearchCV & RandomizedSearchCV
├── 📂 Advance/                 More advanced supervised techniques
├── 📂 Titanic/                 End-to-end Titanic Kaggle workflow
├── 📂 music/                   Music-genre classification using audio features
└── 📂 timepass/                Misc. experiments & quick tests
```

---

## 🎯 Topics by Subfolder

### 📂 `data preprocessing/`
Same content as the top-level `data preprocessing/` folder — categorical encoding using the Titanic dataset and a small GitHub-candidates CSV.

### 📂 `basic knn/`
Classification with `KNeighborsClassifier` plus a notebook that **draws the 2D decision boundary** so you can *see* how `k` affects the model.

### 📂 `basic regression/`
The classic four:
| Subfolder | Algorithm |
|-----------|-----------|
| `diabetes/` | Linear Regression |
| `redge reg/` | Ridge (L2) |
| `lasso reg/` | Lasso (L1) |
| `logistic reg/` | Logistic Regression |

Includes the `visual CV Lreg.py` script that plots 6-fold cross-validation scores with a 95% confidence interval.

### 📂 `Hyperparameter tuning/`
- `GridSearchCV.ipynb` — exhaustive parameter grid search
- `UsingHyperPArarmeterForKnn.ipynb` — tuning `k` and other hyperparameters for KNN
- Uses `diabetes_clean.csv`

### 📂 `Advance/`
A notebook (`code.ipynb`) showcasing more advanced supervised-learning techniques — pipelines, ensembles or model comparison.

### 📂 `Titanic/`
A full mini-project: EDA → preprocessing → model training → Kaggle-format prediction.
Includes both `train.csv` and `test.csv`.

### 📂 `music/`
Predict music **genre** from audio features such as `acousticness`, `danceability`, `energy`, `tempo`, `valence` etc. (`music_clean.csv`).

### 📂 `timepass/`
Quick experiments and scratch work — useful for trying ideas before committing to a clean notebook.

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## ▶️ Where to Start

| Goal | Open this |
|------|-----------|
| Re-cap preprocessing | `data preprocessing/categorical features.ipynb` |
| First classifier | `basic knn/basic knn.ipynb` |
| First regressor | `basic regression/diabetes/code.ipynb` |
| Tune a model | `Hyperparameter tuning/GridSearchCV.ipynb` |
| Full project | `Titanic/Titanic.ipynb` |

---

## 💡 Why a Duplicated Folder?

Many of these subfolders mirror their top-level counterparts on purpose — `supervised/` is curated as a **self-contained supervised-learning bundle**. You can hand someone just this folder and they'd have everything they need.

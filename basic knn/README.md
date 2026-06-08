# 🔵 K-Nearest Neighbors (KNN)

> *"Tell me who your neighbors are, and I'll tell you what you are."*

KNN is the simplest non-parametric classifier — no training phase, just memorize the data and look up the nearest neighbors at prediction time.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `basic knn.ipynb` | End-to-end KNN classification on the student-exam dataset |
| `visualizing_knn.ipynb` | Plot decision boundaries to *see* how KNN classifies |
| `student_exam_dataset.csv` | Tiny dataset — predict pass/fail from hours studied & attendance |

---

## 📊 Dataset — `student_exam_dataset.csv`

A toy dataset perfect for visualization (only 2 features → easy to plot):

| Column | Meaning |
|--------|---------|
| `Hours_Studied` | Hours the student studied |
| `Attendance_Percentage` | Class attendance percentage |
| `Pass` | Target — 1 if passed, 0 if failed |

---

## 🧠 How KNN Works

For a new data point:
1. **Compute distance** to every training point (typically Euclidean).
2. **Pick the `k` closest** neighbors.
3. **Majority vote** — assign the class most common among them.

```
new_point  →  [neighbor₁, neighbor₂, ..., neighborₖ]  →  most common class
```

No model fitting, no parameters learned — it's all in the data.

---

## 🎯 What You'll Learn

### `basic knn.ipynb`
- Loading and splitting the dataset
- Fitting `KNeighborsClassifier(n_neighbors=k)`
- Predicting & scoring with accuracy
- **Choosing `k`:** run KNN for `k = 1..N` and plot the accuracy curve to spot the sweet spot

### `visualizing_knn.ipynb`
- Drawing the **2D decision boundary** with `matplotlib`
- Watching the boundary change as `k` changes:
  - small `k` → jagged, overfits
  - large `k` → smooth, may underfit

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

---

## ▶️ Run

```bash
jupyter notebook "basic knn.ipynb"
jupyter notebook visualizing_knn.ipynb
```

---

## 💡 Tips

- **Scale your features!** KNN uses distance — features with larger ranges will dominate. Use `StandardScaler`.
- **Use odd `k`** for binary classification to avoid ties.
- KNN is **slow at prediction time** for large datasets — every prediction touches the whole training set.
- Try different **distance metrics** (`metric='euclidean'`, `'manhattan'`, `'minkowski'`).

---

🔁 **Next step:** see KNN paired with `GridSearchCV` in [`../Hyperparameter tuning/`](../Hyperparameter%20tuning/).

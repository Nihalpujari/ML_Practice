# 🎯 K-Means Clustering

> Group data points into `k` clusters by minimizing within-cluster distance to the centroid.

This folder contains several K-Means experiments on different datasets — from the classic Iris flowers to GitHub developer profiles and Kaggle dataset metadata.

---

## 📂 Folder Structure

```
KMeans/
│
├── 📂 iris/             K-Means on the Iris flowers dataset (k=3)
├── 📂 github/           Cluster GitHub developer profiles
├── 📂 Kaggle/           Cluster Kaggle datasets by their metadata
└── iris_data.ipynb      Standalone Iris notebook (same as iris/ but at top level)
```

---

## 🧠 The K-Means Algorithm

1. **Initialize** `k` cluster centroids (often via `k-means++`)
2. **Assign** each point to the nearest centroid → cluster membership
3. **Update** centroids: new centroid = mean of its members
4. **Repeat** until centroids stop moving (or you hit `max_iter`)

The objective minimized:

```
Inertia = Σ (distance from point to its centroid)²
```

---

## 🔍 Subfolder Tour

### 📂 `iris/`
Cluster Iris flowers using their sepal & petal measurements.
- `iris_data.ipynb` — load Iris, scale features, run K-Means with `k=3`, plot clusters.
- **Validation:** compare clusters to actual species labels (purity check).

### 📂 `github/`
Cluster GitHub user profiles. Dataset: `github_candidates_1.csv`.
- Features: `public_repos`, `total_stars`, `total_forks`, `commits_12m`, `dominant_language`, `has_cicd`.
- 💡 Discover archetypes — "starred-but-quiet" devs vs "high-activity" devs etc.

### 📂 `Kaggle/`
Cluster Kaggle dataset metadata. Dataset: `kaggle-preprocessed.csv`.
- Features include `No_of_files`, `size`, `Type_of_file`, `Upvotes`, `Medals`, `Usability`.
- Two notebook variants (`code.ipynb`, `ewrty.ipynb`) explore different feature choices and `k` values.

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## ▶️ Picking `k` — The Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
for k in range(1, 11):
    inertias.append(KMeans(n_clusters=k, random_state=42).fit(X).inertia_)

plt.plot(range(1, 11), inertias, 'o-')
plt.xlabel('k'); plt.ylabel('inertia')
plt.title('Elbow Method')
```

Pick the `k` at the "elbow" — where adding more clusters stops paying off.

---

## 💡 Tips

- **Scale features first** with `StandardScaler` — K-Means uses Euclidean distance.
- Run K-Means **multiple times** with different `random_state` to confirm stability.
- Use the **Silhouette score** (`sklearn.metrics.silhouette_score`) for a more rigorous `k` choice.

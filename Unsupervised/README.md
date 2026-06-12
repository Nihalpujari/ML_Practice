# 🔍 Unsupervised Learning

> **No labels? No problem.**
> Unsupervised learning finds structure hidden inside raw data — groupings, projections, anomalies — without anyone telling the model what's "right".

This folder explores **K-Means clustering** across multiple datasets and a fully-worked **Penguin species clustering** project that combines PCA + clustering.

---

## 📂 Folder Structure

```
Unsupervised/
│
├── 📂 KMeans/
│   ├── 📂 iris/         K-Means on the classic Iris dataset
│   ├── 📂 github/       Clustering GitHub user profiles
│   ├── 📂 Kaggle/       Clustering Kaggle datasets
│   └── iris_data.ipynb  (standalone Iris notebook)
│
└── 📂 Clustering Antarctic Penguin Species/
    ├── code.ipynb       Full PCA + K-Means pipeline
    └── README.md        Detailed project README ✨
```

---

## 🎯 Topics

### 1. K-Means Clustering
Partitions data into `k` non-overlapping clusters by minimizing within-cluster variance.

The algorithm:
1. Randomly initialize `k` centroids
2. Assign each point to the nearest centroid
3. Recompute centroids as the mean of assigned points
4. Repeat steps 2–3 until centroids stop moving

📌 You **choose `k`** — typically via the **Elbow Method** (plot inertia vs `k` and pick the bend).

### 2. PCA (Principal Component Analysis)
Reduces high-dimensional data to a few orthogonal components that capture the most variance. Used both for:
- **Visualization** — squash to 2D to plot clusters
- **Speed-up** — fewer features → faster clustering

---

## 📁 Subfolders Explained

### 📂 `KMeans/iris/`
The Hello World of clustering. The Iris dataset has 3 species — `K=3` clusters should naturally emerge from the petal/sepal measurements alone.

### 📂 `KMeans/github/`
Cluster GitHub developer profiles using `github_candidates_1.csv` — features like `public_repos`, `total_stars`, `commits_12m`, `has_cicd`. Discover natural "developer archetypes".

### 📂 `KMeans/Kaggle/`
Cluster Kaggle dataset metadata using `kaggle-preprocessed.csv`:
- `Dataset_name, Author_name, No_of_files, size, Type_of_file, Upvotes, Medals, Usability`
- Multiple notebooks (`code.ipynb`, `ewrty.ipynb`) try different angles

### 📂 `Clustering Antarctic Penguin Species/`
A **complete end-to-end project**:
1. Load the Palmer Penguins dataset
2. Preprocess (drop NaNs, encode `sex`)
3. Standardize features
4. Apply PCA for dimensionality reduction
5. Run K-Means and visualize the clusters
6. Validate against the (hidden) species labels

📄 See its **dedicated README** for the full walkthrough.

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 💡 Tips

- **Always scale features** before clustering — distance-based algorithms are sensitive to feature magnitudes.
- The **Elbow Method** (`plot(k, inertia)`) is a quick way to choose `k`. The **Silhouette score** is a more rigorous alternative.
- K-Means assumes **spherical, equally-sized clusters** — if your data isn't shaped that way, try DBSCAN or Gaussian Mixture Models.
- Use `random_state=42` for reproducibility — K-Means initialization is random.

---

🔁 **Next step:** apply tuned models on real data — [`../Titanic/`](../Titanic/) or [`../real data/`](../real%20data/).

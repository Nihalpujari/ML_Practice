# 🐧 Penguin Clustering Project (Unsupervised Learning)

## 📌 Overview
This project focuses on analyzing and clustering penguin data using **unsupervised machine learning techniques**. The goal is to group penguins based on their physical characteristics without using labeled species data.

We use **PCA (Principal Component Analysis)** for dimensionality reduction and clustering algorithms to identify patterns in the dataset.

---

## 📊 Dataset
The dataset used is the **Palmer Penguins Dataset**, which contains measurements of penguins in Antarctica.

### Features:
- `culmen_length_mm` – Length of the bill
- `culmen_depth_mm` – Depth of the bill
- `flipper_length_mm` – Flipper length
- `body_mass_g` – Body mass
- `sex` – Gender of penguin

---

## ⚙️ Technologies Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn (Pipeline, PCA, Clustering)

---

## 🚀 Project Workflow

### 1. Data Preprocessing
- Handle missing values (NaN removal)
- Convert categorical data to numerical (if required)
- Feature scaling

### 2. Pipeline Creation
A pipeline is used to combine multiple steps:
- StandardScaler
- PCA (Dimensionality Reduction)
- Clustering Algorithm (like KMeans)

```python
from sklearn.pipeline import make_pipeline

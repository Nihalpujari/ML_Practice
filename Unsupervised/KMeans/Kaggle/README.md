# 📦 K-Means on Kaggle Dataset Metadata

Cluster Kaggle datasets themselves — by their popularity, size, type and usability score.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `code.ipynb` | Main K-Means pipeline on Kaggle metadata |
| `ewrty.ipynb` | Alternative version — different feature choices / `k` values |
| `kaggle-preprocessed.csv` | Cleaned metadata of multiple Kaggle datasets |

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `Dataset_name` | Title of the Kaggle dataset |
| `Author_name` / `Author_id` | Uploader info |
| `No_of_files` | Number of files in the dataset |
| `size` | Dataset size (e.g. `491 kB`) |
| `Type_of_file` | `CSV`, `JSON`, etc. |
| `Upvotes` | Number of upvotes |
| `Medals` | `Bronze`, `Silver`, `Gold`… |
| `Usability` | Kaggle's auto-computed usability score |
| `Date` / `Day` / `Time` | Upload timestamp |
| `Dataset_link` | URL to the original dataset |

---

## 🎯 What You'll Learn

1. Clean mixed-format columns — e.g. parse `size` from `"491 kB"` into a numeric value
2. Encode categorical columns (`Type_of_file`, `Medals`)
3. Scale numeric features
4. Apply K-Means and explore clusters such as:
   - 🥇 *"High-upvote, gold-medal"* power datasets
   - 📊 *"Small, niche CSVs"*
   - 📚 *"Mega-files, low engagement"*

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 💡 Tip

The two notebooks (`code.ipynb` and `ewrty.ipynb`) explore the same problem with different choices — keep them open side-by-side to see how feature selection changes the cluster shape.

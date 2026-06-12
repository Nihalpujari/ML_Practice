# 🐙 K-Means on GitHub User Profiles

Cluster developers by their GitHub activity — discover natural "archetypes" without ever labeling anyone.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `code.ipynb` | Load → preprocess → K-Means → visualize developer clusters |
| `github_candidates_1.csv` | Small dataset of 10 GitHub profiles |

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `username` | GitHub handle (drop before clustering) |
| `public_repos` | Number of public repositories |
| `total_stars` | Sum of stars across repos |
| `total_forks` | Sum of forks across repos |
| `commits_12m` | Commits made in the last 12 months |
| `dominant_language` | Most-used language (`Ruby`, `C`, `Python`, …) |
| `has_cicd` | Binary — uses CI/CD pipelines? |

---

## 🎯 What You'll Learn

1. Drop the identifier column (`username`)
2. Encode `dominant_language` (label or one-hot)
3. Scale all numeric features with `StandardScaler`
4. Use the **Elbow Method** to find a sensible `k`
5. Inspect each cluster's centroid to label them:
   - 🌟 "Mega-popular but inactive"
   - 🛠️ "High commits, low stars"
   - 🆕 "Newbie / low activity"

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 💡 Tip

With only 10 rows, K-Means is more for **practicing the workflow** than producing statistically meaningful clusters. The same code would scale to millions of profiles.

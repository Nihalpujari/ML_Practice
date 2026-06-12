# 🌍 Real Data — Stack Overflow Profiles

> Toy datasets are clean. **Real datasets are not.**
> This folder works with scraped Stack Overflow user data so you can practice EDA on real, messy-ish features.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `stackoverflow.ipynb` | Explore, visualize and model the Stack Overflow user dataset |
| `stackoverflow_200.csv` | 200 top users scraped from Stack Overflow |

---

## 📊 Dataset

Each row is one Stack Overflow user:

| Column | Description |
|--------|-------------|
| `user_id` | SO user ID |
| `display_name` | Public username |
| `profile_link` | URL to the user's profile |
| `reputation` | Total reputation score |
| `answers_count_fetched` | Number of answers fetched (capped at 200) |
| `accepted_answers_fetched` | How many of those were accepted |
| `accepted_answer_ratio` | `accepted / total` |
| `total_answer_score_fetched` | Sum of upvotes across fetched answers |
| `avg_score_per_answer` | Average upvotes per answer |
| `top_tags` | Top tags with answer counts, e.g. `python:12345(456)` |

---

## 🎯 What You'll Learn

### Exploratory Data Analysis
- Distribution of **reputation** (very long-tailed!)
- Relationship between **acceptance ratio** and **average score**
- Top tags overall — which technologies dominate Stack Overflow

### Real-World Preprocessing
- The `top_tags` column is a **messy string** like:
  ```
  datetime:43332(10223), c++:198658(30000), textview:56191(6986)
  ```
  Parse it into a usable structure (e.g. dict, set of tags, or one-hot vector).
- The `size` and `reputation` columns are heavily skewed — try a **log transform**.

### Modelling Ideas
- Predict `accepted_answer_ratio` from the other features (regression)
- Cluster users by activity profile (unsupervised)
- Build a tag-recommendation prototype

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## ▶️ Run

```bash
jupyter notebook stackoverflow.ipynb
```

---

## 💡 Tips

- The dataset is **biased toward power users** (top 200) — don't generalize to "all SO users".
- `reputation` follows a **power law** — always plot it on a log scale.
- The `top_tags` parsing is great practice for regex / string-splitting skills.
- For modelling, drop or hash high-cardinality columns like `display_name` and `profile_link`.

---

🔁 **Related:** for another scraped real-world dataset, see [`../Unsupervised/KMeans/Kaggle/`](../Unsupervised/KMeans/Kaggle/).

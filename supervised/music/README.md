# 🎵 Music Genre Classification

> Predict the **genre** of a track from its audio features.
> A fun, beginner-friendly supervised-classification problem.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `code.ipynb` | Train & evaluate a classifier on audio features |
| `music_clean.csv` | Cleaned dataset of tracks with audio attributes |

---

## 📊 Dataset

Each row is a track. Features include:

| Feature | Description |
|---------|-------------|
| `popularity` | Spotify popularity score (0–100) |
| `acousticness` | Confidence the track is acoustic |
| `danceability` | How suitable for dancing |
| `duration_ms` | Track length in milliseconds |
| `energy` | Perceptual intensity & activity |
| `instrumentalness` | Likelihood of being instrumental |
| `liveness` | Probability of a live audience |
| `loudness` | Loudness in dB |
| `speechiness` | Spoken-word presence |
| `tempo` | BPM |
| `valence` | Musical positivity / happiness |
| `genre` | 🎯 Target — the genre label |

---

## 🎯 What You'll Learn

1. Inspect a real-world multi-class classification dataset
2. Preprocess (scale numeric features, handle the unnamed index column)
3. Fit and compare classifiers (Logistic Regression, KNN, Random Forest, etc.)
4. Evaluate with **confusion matrix** + per-class precision/recall

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## ▶️ Run

```bash
jupyter notebook code.ipynb
```

---

## 💡 Tips

- The CSV starts with an **unnamed index column** — drop it (`df.drop(df.columns[0], axis=1)`) before training.
- **Scale features** — `loudness` and `duration_ms` are on very different ranges from `valence` or `energy`.
- Genres can be unbalanced — check `df['genre'].value_counts()` and use **stratified split** if needed.

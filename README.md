# 🤖 ML Practice

A hands-on collection of **Machine Learning notebooks** covering the full spectrum of the ML workflow — from raw data preprocessing all the way through supervised learning, unsupervised learning, NLP, and hyperparameter tuning.

> Built with Python & Jupyter Notebooks as a personal learning and reference repository.

---

## 📁 Repository Structure

```
ML_Practice/
├── data preprocessing/        # Cleaning, encoding, scaling raw datasets
├── basic regression/          # Linear & polynomial regression fundamentals
├── basic knn/                 # K-Nearest Neighbors implementation & intuition
├── supervised/                # Classification & regression with sklearn
├── unsupervised/              # Clustering & dimensionality reduction
├── Hyperparameter tuning/     # GridSearchCV, RandomizedSearchCV, cross-validation
├── NLP/                       # Text preprocessing, vectorization & classification
├── Titanic/                   # End-to-end ML project on the Titanic dataset
├── real data/                 # Practice on real-world datasets
└── all.ipynb                  # Consolidated notebook with multiple experiments
```

---

## 📚 Topics Covered

### 🧹 Data Preprocessing
- Handling missing values (imputation strategies)
- Feature encoding (Label Encoding, One-Hot Encoding)
- Feature scaling (StandardScaler, MinMaxScaler)
- Train/test splitting

### 📈 Basic Regression
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Model evaluation with R², MAE, MSE, RMSE

### 🔵 K-Nearest Neighbors (KNN)
- Distance metrics (Euclidean, Manhattan)
- Choosing optimal `k` values
- KNN for classification and regression tasks

### 🧠 Supervised Learning
- Logistic Regression
- Decision Trees & Random Forests
- Support Vector Machines (SVM)
- Naive Bayes
- Gradient Boosting
- Model evaluation: confusion matrix, accuracy, precision, recall, F1-score

### 🔍 Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Visualizing clusters and explained variance

### ⚙️ Hyperparameter Tuning
- GridSearchCV
- RandomizedSearchCV
- K-Fold Cross Validation
- Pipeline construction with sklearn

### 🗣️ Natural Language Processing (NLP)
- Text cleaning & preprocessing
- Bag of Words (BoW) and TF-IDF vectorization
- Sentiment analysis
- Text classification with ML models

### 🚢 Titanic (End-to-End Project)
- Exploratory Data Analysis (EDA)
- Feature engineering
- Building and comparing multiple models
- Kaggle-style submission pipeline

### 📊 Real Data
- Applying ML techniques to real-world datasets
- Exploratory analysis, feature selection, and model building

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3** | Core language |
| **Jupyter Notebook** | Interactive development environment |
| **pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **scikit-learn** | ML algorithms, preprocessing, evaluation |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical data visualization |
| **NLTK / re** | NLP text processing |

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.7+ and pip installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Nihalpujari/ML_Practice.git
cd ML_Practice
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter nltk
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

Then navigate to any folder and open the `.ipynb` file of your choice.

---

## 📖 How to Use This Repo

- **Beginners:** Start with `data preprocessing/` → `basic regression/` → `basic knn/` to build foundational intuition.
- **Intermediate learners:** Explore `supervised/`, `unsupervised/`, and `Hyperparameter tuning/` for a broader ML toolkit.
- **NLP enthusiasts:** Head to `NLP/` for text processing and classification workflows.
- **Project reference:** The `Titanic/` folder demonstrates a complete end-to-end ML project workflow.
- **Quick overview:** Open `all.ipynb` for a combined look at multiple experiments in one place.

---

## 🗺️ Learning Roadmap

```
Data Preprocessing
      ↓
Basic Regression & KNN
      ↓
Supervised Learning (Classification & Regression)
      ↓
Unsupervised Learning (Clustering & PCA)
      ↓
Hyperparameter Tuning & Cross Validation
      ↓
NLP (Text Classification & Sentiment Analysis)
      ↓
End-to-End Projects (Titanic, Real Data)
```

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

1. Fork this repository
2. Create a new branch: `git checkout -b feature/your-topic`
3. Commit your changes: `git commit -m "Add notebook on XYZ"`
4. Push to the branch: `git push origin feature/your-topic`
5. Open a Pull Request

---

## 📬 Contact

**Nihal Pujari**
GitHub: [@Nihalpujari](https://github.com/Nihalpujari)

---

## 📄 License

This project is open-source and available for personal learning and educational use.

---

> ⭐ If you find this repository useful, consider giving it a star!

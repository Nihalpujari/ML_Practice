import matplotlib.pyplot as plt
import numpy as np

# Your CV scores
cv_scores = np.array([0.74451678, 0.77241887, 0.76842114, 0.7410406, 0.75170022, 0.74406484])

# Compute mean and SD
mean_score = np.mean(cv_scores)
sd_score = np.std(cv_scores)
n = len(cv_scores)

# Compute 95% confidence interval
se = sd_score / np.sqrt(n)  # Standard error
ci = 1.96 * se              # 95% CI

# Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, n+1), cv_scores, 'o-', label='CV Scores', color='blue')
plt.axhline(mean_score, color='green', linestyle='--', label=f'Mean R² = {mean_score:.3f}')
plt.fill_between(range(1, n+1), mean_score-ci, mean_score+ci, color='green', alpha=0.2, label='95% CI')
plt.xticks(range(1, n+1))
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('6-Fold Cross-Validation Scores with Mean and 95% CI')
plt.legend()
plt.show()
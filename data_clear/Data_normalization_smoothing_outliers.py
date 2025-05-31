import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set working directory (optional, you can set path to your file directly)
# os.chdir("E:/PXX/R Script/Hydroinfor_HW/HW2")  # Not necessary in Python

# Load the data
df = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/Sediment.csv")

# Check data format (like str() or typeof() in R)
print(df.info())  # For column types and null counts

# 1.1 Normalize the data to the range [0, 1] using different methods

# Min-Max Normalization
def min_max(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
# self-define function 1)

# Apply min-max normalization
min_max_df = df.apply(min_max)

# Z-Score Normalization
def z_score(x):
    return (x - np.mean(x)) / np.std(x)

# Apply z-score normalization
z_score_df = df.apply(z_score)

# Decimal Normalization
def round_up(x):
    return 10 ** np.ceil(np.log10(x))

def decimal_norm(x):
    return x / round_up(np.max(x))

# Apply decimal normalization
decimal_norm_df = df.apply(decimal_norm)


# 1.2 Plot Scatterplot for Min-Max vs Decimal Normalization
scatter_df = pd.DataFrame({'min_max': min_max_df['theta'], 'decimal': decimal_norm_df['theta']})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=scatter_df, x='min_max', y='decimal', color='blue', s=50)
plt.title('Scatter plot for Min-Max Method vs Decimal Method')
plt.xlabel('Min-Max Method')
plt.ylabel('Decimal Method')
# plt.show()


# 1.3 Boxplot for all the variables (including outliers detection)
# Boxplot stats to detect outliers
outliers = pd.DataFrame({
    'min_max_theta_outliers': min_max_df['theta'][min_max_df['theta'] < (np.percentile(min_max_df['theta'], 25) - 1.5 * (np.percentile(min_max_df['theta'], 75) - np.percentile(min_max_df['theta'], 25)))],
    'min_max_theta_upper_outliers': min_max_df['theta'][min_max_df['theta'] > (np.percentile(min_max_df['theta'], 75) + 1.5 * (np.percentile(min_max_df['theta'], 75) - np.percentile(min_max_df['theta'], 25)))]
})
print(outliers)

# Boxplot for min-max normalized data
plt.figure(figsize=(8, 6))
sns.boxplot(data=min_max_df, orient='v', palette="Set2")  # 'v' for vertical orientation
plt.title('Boxplot for Min-Max Normalized Data with Outliers')
plt.xlabel('Variable')
plt.ylabel('Value')
plt.show()


# 1.4 Smooth Data using LOESS (Local Polynomial Regression)
# Scatter plot for theta vs theta_p
theta1 = min_max_df['theta']
theta_p1 = min_max_df['theta_p']

plt.figure(figsize=(8, 6))
plt.scatter(theta1, theta_p1, c='blue', label='Data points')
plt.title('Scatter plot of Theta vs Theta_p')
plt.xlabel('Theta')
plt.ylabel('Theta_p')
# plt.show()

# LOESS using LOWESS (Locally Weighted Scatterplot Smoothing)
from statsmodels.nonparametric.smoothers_lowess import lowess

# Smooth the data
smoothed = lowess(theta_p1, theta1, frac=0.5)  # frac controls the smoothness

# Plot smoothed data
plt.figure(figsize=(8, 6))
plt.scatter(theta1, theta_p1, c='blue', label='Data points')
plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOESS smooth curve')
plt.title('Scatter plot of Theta vs Theta_p with LOESS smoothing')
plt.xlabel('Theta')
plt.ylabel('Theta_p')
plt.legend()
# plt.show()

# If you want a quadratic fit (not necessarily better, but more flexible):
from numpy import polyfit

# Fit a quadratic curve
coeffs = polyfit(theta1, theta_p1, 2)
quad_fit = np.polyval(coeffs, theta1)

# Plot quadratic fit
plt.figure(figsize=(8, 6))
plt.scatter(theta1, theta_p1, c='blue', label='Data points')
plt.plot(theta1, quad_fit, color='green', label='Quadratic fit', linestyle='--')
plt.title('Scatter plot of Theta vs Theta_p with Quadratic Fit')
plt.xlabel('Theta')
plt.ylabel('Theta_p')
plt.legend()
plt.show()

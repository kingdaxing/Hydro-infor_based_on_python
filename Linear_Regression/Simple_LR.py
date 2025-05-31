import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1.1 Scatterplot Matrices
# Load data (make sure the file path is correct)
sediment = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/Sediment.csv")

# M1: Scatterplot matrix (pairplot in Python)
sns.pairplot(sediment)
plt.show()

# M2: You can directly use seaborn pairplot for the same effect.
# sns.pairplot(sediment)

# M3: Alternatively, for detailed scatterplot matrix with correlation coefficients
sns.pairplot(sediment, kind="scatter", plot_kws={'alpha':0.5})
plt.show()

# M4: For correlation matrix using Pearson
cor_matrix = sediment.corr(method="pearson")
print(cor_matrix)


#1.2 Simple Linear Regression
import statsmodels.api as sm

# Assuming 'theta' and 'theta_p' are columns in the 'sediment' dataframe
# Prepare the data for regression
X = sediment['theta_p']
y = sediment['theta']

# Add a constant for the intercept term
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# To access specific parts of the model
print("Coefficients:", model.params)
print("Residuals:", model.resid)

# Prediction
new_data = pd.DataFrame({'theta_p': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]})
new_data = sm.add_constant(new_data)  # Add constant for intercept term
predictions = model.predict(new_data)
print("Predicted values:", predictions)

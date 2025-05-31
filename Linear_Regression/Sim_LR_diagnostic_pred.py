# 1.3 diagnostic plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

sediment = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/Sediment.csv")

# Fit a simple linear regression model
X = sediment['theta_p']
y = sediment['theta']
X = sm.add_constant(X)  # Add constant to the model (intercept term)
model = sm.OLS(y, X).fit()

# Print summary of the regression
print(model.summary())
# Extracting specific values:
p_value = model.pvalues[1]  # p-value for the slope (theta_p)
r_squared = model.rsquared  # R-squared value
# Print the p-value and R-squared value
print(f"P-value: {p_value}")
print(f"R-squared: {r_squared}")


# Get fitted values and residuals
fitted_values = model.fittedvalues
residuals = model.resid

# Plot the residuals vs. fitted values
sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Residuals vs. Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Residuals vs. fitted values plot
axes[0, 0].scatter(fitted_values, residuals)
axes[0, 0].axhline(0, color='gray', linestyle='--')
axes[0, 0].set_title("Residuals vs Fitted")
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")

# Normal Q-Q plot for residuals
sm.qqplot(residuals, line ='45', ax=axes[0, 1])

# Histogram of residuals
axes[1, 0].hist(residuals, bins=30, edgecolor='black')
axes[1, 0].set_title("Histogram of Residuals")

# Scale-Location plot (sqrt of standardized residuals)
axes[1, 1].scatter(fitted_values, np.sqrt(np.abs(residuals)))
axes[1, 1].axhline(0, color='gray', linestyle='--')
axes[1, 1].set_title("Scale-Location Plot")
axes[1, 1].set_xlabel("Fitted Values")
axes[1, 1].set_ylabel("Sqrt(|Residuals|)")

plt.tight_layout()
plt.show()



# 1.4 Prediction with Confidence and Prediction Intervals
# Predict with confidence intervals and prediction intervals
new_data = pd.DataFrame({'theta_p': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]})
new_data = sm.add_constant(new_data)  # Add constant for intercept term

# Confidence interval and prediction interval
predictions = model.get_prediction(new_data)
summary_frame = predictions.summary_frame(alpha=0.05)  # 95% CI
print(summary_frame)

# Plotting the predicted values with confidence and prediction intervals
plt.figure(figsize=(10, 6))
plt.plot(new_data['theta_p'], summary_frame['mean'], color='blue', label='Predicted')
plt.fill_between(new_data['theta_p'], summary_frame['obs_ci_lower'], summary_frame['obs_ci_upper'], color='lightblue', alpha=0.5, label='Prediction Interval')
plt.fill_between(new_data['theta_p'], summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], color='lightgreen', alpha=0.5, label='Confidence Interval')
plt.title('Prediction and Confidence Intervals')
plt.xlabel('theta_p')
plt.ylabel('Predicted theta')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# matplotlib: https://matplotlib.org/stable/plot_types/arrays/pcolormesh.html
import seaborn as sns
import plotly.express as px
from datetime import datetime
from scipy.stats import kurtosis, skew
from statsmodels import api as sm


# 3.1: Plot as time series using pyplot
# Load time series data
df_Q3 = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/otchybrid.csv")
# Convert 'date' to datetime
df_Q3.columns = ['date'] + list(df_Q3.columns[1:])
df_Q3['date'] = pd.to_datetime(df_Q3['date'], format="%m/%d/%y %H:%M")
# print(df_Q3['date'].head())

# Plot time series using pyplot
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(df_Q3['date'], df_Q3['Residual'], label='Residual', color='b', linewidth=2)
# Internally, matplotlib uses *args to handle an arbitrary number of arguments.

# Format the x-axis (date labels)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # Set month breaks (adjust as needed)

# ?! Other requirement in this Q: Set 2-month interval breaks
# Sample data for demonstration (replace with your own data)
# dates = pd.date_range('2023-01-01', periods=12, freq='M')
# values = range(12)
#
# plt.figure(figsize=(10, 6))
# plt.plot(dates, values)
#
# # Set 2-month interval breaks on x-axis
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Set 2-month interval breaks
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format the ticks to show month and year
#
# # Rotate the tick labels for better readability
# plt.xticks(rotation=45)

plt.gcf().autofmt_xdate()  # Rotate x-axis labels for readability
plt.xlabel('Date')
plt.ylabel('Residual')
plt.title('Time Series of Residual')

plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to fit all elements
# plt.show()


# 3.2: Scatterplot (E.W.M21 vs N.S.M21)
plt.figure(figsize=(8, 6))  # Optional: set figure size

# note: the head in csv cannot be extracted into py correctly!
# Step 1:
# Clean up the column names by removing spaces
df_Q3.columns = df_Q3.columns.str.strip()  # Remove any leading/trailing spaces
# Optionally, replace spaces with underscores for easier referencing
df_Q3.columns = df_Q3.columns.str.replace(' ', '_')

# Check the updated column names
print(df_Q3.columns)

# Now, plot the scatterplot using the cleaned column names
sns.scatterplot(data=df_Q3, x='E-W_M21', y='N-S_M21', color='blue', s=20)

# Step 2: Set plot title and labels
plt.title('Scatter plot for E.W.M21 versus N.S.M21')
plt.xlabel('E.W.M21')
plt.ylabel('N.S.M21')

# Optional: Add grid for better readability
plt.grid(True)

# Step 3: Show the plot
plt.tight_layout()  # Adjust layout to fit all elements
# plt.show()


# 3.3: Histograms for Ave.E.W, Ave.N.S, and AveSpeed
# Set up a 3x1 grid for subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

# Plot 1: Histogram of 'Ave.E.W'
sns.histplot(df_Q3['Ave_E-W'], kde=False, color='lightblue', bins=20, ax=axes[0])
axes[0].set_title('Histogram of Ave.E.W')
axes[0].set_xlabel('Ave.E.W')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Add gridlines to the first plot


# Plot 2: Histogram of 'Ave.N.S'
sns.histplot(df_Q3['Ave_N-S'], kde=False, color='lightgreen', bins=20, ax=axes[1])
axes[1].set_title('Histogram of Ave.N.S')
axes[1].set_xlabel('Ave.N.S')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Add gridlines to the second plot


# Plot 3: Histogram of 'AveSpeed'
sns.histplot(df_Q3['AveSpeed'], kde=True, color='lightyellow', bins=20, ax=axes[2])
axes[2].set_title('Histogram of AveSpeed')
axes[2].set_xlabel('AveSpeed')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Add gridlines to the third plot

# Adjust layout for better spacing
plt.tight_layout()
# Optionally, adjust spacing further
plt.subplots_adjust(hspace=0.6)  # Increase the vertical space between subplots
# plt.show()


# 3.4: Boxplots
df_Q3_melted = pd.melt(df_Q3[['Ave_E-W', 'Ave_N-S', 'AveSpeed']], var_name='Group', value_name='Value')
sns.boxplot(x='Group', y='Value', data=df_Q3_melted)
plt.title('Box Plot of Ave.E.W, Ave.N.S, and AveSpeed')
plt.show()


# 3.5: Minimal line plot (scatter with rug)
# 3.5.1 Dot-dash plot & Scatter plot
# view 'seaborn' package: https://seaborn.pydata.org/generated/seaborn.displot.html
sns.scatterplot(data=df_Q3, x='Ave_E-W', y='Ave_N-S', color='blue', s=30)
# Plot the rug plot for Ave.E-W (along the x-axis)
sns.rugplot(data=df_Q3, x='Ave_E-W', color='black', height=0.05, linewidth=1)
sns.rugplot(data=df_Q3, y='Ave_N-S', color='black', height=0.05, linewidth=1)

# Customizing the plot for a clean and minimal design
plt.title('Scatter plot for Ave.E.W versus Ave.N.S')
plt.xlabel('Ave.E.W')
plt.ylabel('Ave.N.S')

# Default ticks for a minimal design
plt.xticks()
plt.yticks()

# Display the plot
plt.tight_layout()  # To make sure everything fits
plt.show()


# 3.5.2: Marginal histogram scatterplot
sns.jointplot(x='Ave_E-W', y='Ave_N-S', data=df_Q3, kind='scatter', marginal_kws={'color': 'yellow'})
plt.show()

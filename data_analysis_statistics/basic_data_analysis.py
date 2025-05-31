import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from scipy.stats import kurtosis, skew
from statsmodels import api as sm

# 1.1: Load data
my_data = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/Sediment.csv")
# if excel, pd.read_excel

# Check data structure (equivalent of R's str())
print(my_data.info())

# 1.2: Indexing the dataframe (Pythonic way)
# Selecting rows 11-50 (index starts at 0, so 10:50 includes rows 11 to 50). Format[Row, Column]
my_data_new = my_data.iloc[10:49, :]
# iloc[]: integer-location-based indexing
my_data_new.to_csv("my_data_new.txt", index=False)

# 1.3: Calculate the percentage of values above 1.5 in 'theta' column
count_above_1_5 = (my_data_new['theta'] > 1.5).sum()  # Boolean indexing and summing True values
total_count = len(my_data_new)
# if courting, num_columns = len(my_data_new.columns) or = my_data_new.shape[1]
percentage_above_1_5 = count_above_1_5 / total_count
print(f"Percentage of values above 1.5 in 'theta' column: {percentage_above_1_5:.4f}")

# Alternative calculation of percentage - Boolean array
percentage_1 = (my_data_new['theta'] > 1.5).mean()  # Directly using mean of boolean array
print(f"Alternative Percentage: {percentage_1:.4f}")


# 2.1: Statistical properties (mean, median, variance, etc.)
# Mean for each column
column_means = my_data.mean(skipna=True)
print(f"Column means:\n{column_means}")

# Median for each column
column_median = my_data.median(skipna=True)
print(f"Column medians:\n{column_median}")

# Mode for 'theta' column (most frequent value)
mode_theta = my_data['theta'].mode()[0]
print(f"Mode of 'theta': {mode_theta}")


# 2.2
# # Variance
# variance = my_data_new['theta'].var()
#
# # Standard Deviation
# std_dev = my_data_new['theta'].std()
#
# # Interquartile Range (IQR)
# IQR = my_data_new['theta'].quantile(0.75) - my_data_new['theta'].quantile(0.25)
#
# # Skewness
# skewness = my_data_new['theta'].skew()
#
# # Kurtosis
# kurtosis = my_data_new['theta'].kurt() # correct import 'from... as kurt'



# Variance and standard deviation
column_var = my_data.var(skipna=True)
column_sds = my_data.std(skipna=True)
print(f"Column variances:\n{column_var}")
print(f"Column standard deviations:\n{column_sds}")

# Skewness and Kurtosis
column_skew = my_data.skew(skipna=True)
column_kurt = my_data.apply(lambda x: kurtosis(x, nan_policy='omit'))
print(f"Column skewness:\n{column_skew}")
print(f"Column kurtosis:\n{column_kurt}")

# 2.2: Descriptive statistics using pandas describe() - this already provides counts, mean, std, min, max, etc.
data_properties = my_data.describe()
print(data_properties)



# 2.3: Handling missing data
my_data2 = my_data.copy()
my_data2.iloc[[0, 4, 14]] = np.nan  # Introducing NaNs to simulate missing data
# if in column: my_data2.iloc[:, [0, 4, 14]] = np.nan


# Recalculate with NaNs ignored
column_means2 = my_data2.mean(skipna=True)
column_sds2 = my_data2.std(skipna=True)
print(f"New means (with missing data): {column_means2}")
print(f"New standard deviations (with missing data): {column_sds2}")

# 2.1 Data Loading and Processing (for Global Temperature Anomaly)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # For proper datetime formatting

# Load data
global_temp = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/global_monthly_mean_temperature_anomaly.csv")

# Inspect the first few rows to understand the structure
print(global_temp.head())

# Filter for specific data sources (GCAG and GISTEMP)
GCAG = global_temp[global_temp['Source'] == 'GCAG']
GIS = global_temp[global_temp['Source'] == 'GISTEMP']

# Order data by 'Date'
GCAG_new = GCAG.sort_values('Date')
GIS_new = GIS.sort_values('Date')

# Extract all January data
GCAG_new['Date'] = pd.to_datetime(GCAG_new['Date'])
GIS_new['Date'] = pd.to_datetime(GIS_new['Date'])

GCAG_Jan_data = GCAG_new[GCAG_new['Date'].dt.month == 1]
GIS_Jan_data = GIS_new[GIS_new['Date'].dt.month == 1]

# Combine the data
GCAG_GIS_Jan_data = pd.concat([GCAG_Jan_data, GIS_Jan_data])

# Plot the data (using line plot for time series)
plt.figure(figsize=(10, 6))
sns.lineplot(data=GCAG_GIS_Jan_data, x='Date', y='Mean', hue='Source', style='Source')
plt.title("Temperature Anomaly in January: GCAG and GISTEMP")
plt.xlabel('Time')
plt.ylabel('Monthly Mean Anomaly Temp (℃)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 2.2 Annual Mean Temperature (excluding 2016)
# Filter out the 2016 data (from January to May)
filter_GCAG = GCAG_new[~GCAG_new['Date'].dt.strftime('%Y-%m').str.match(r'^2016-(0[1-5])')]
filter_GIS = GIS_new[~GIS_new['Date'].dt.strftime('%Y-%m').str.match(r'^2016-(0[1-5])')]

# Extract the 'Year' from the 'Date' column
filter_GCAG['Year'] = filter_GCAG['Date'].dt.year
filter_GIS['Year'] = filter_GIS['Date'].dt.year

# Calculate the annual mean for GCAG
annual_mean_GCAG = filter_GCAG.groupby('Year')['Mean'].mean().reset_index()
annual_mean_GCAG.columns = ['Year', 'Mean_GCAG']

# Calculate the annual mean for GIS
annual_mean_GIS = filter_GIS.groupby('Year')['Mean'].mean().reset_index()
annual_mean_GIS.columns = ['Year', 'Mean_GIS']

# Merge the two datasets on 'Year'
annual_mean = pd.merge(annual_mean_GCAG, annual_mean_GIS, on='Year')

# Plot the annual mean temperature for GCAG and GISTEMP
plt.figure(figsize=(12, 6))
sns.lineplot(data=annual_mean, x='Year', y='Mean_GCAG', label='GCAG', color='blue', linewidth=2)
sns.lineplot(data=annual_mean, x='Year', y='Mean_GIS', label='GISTEMP', color='red', linewidth=2)
plt.title("Annual Mean Temperature: GCAG and GISTEMP")
plt.xlabel('Year')
plt.ylabel('Annual Mean Temperature (℃)')
plt.legend(title='Sources')
plt.xticks(ticks=np.arange(1880, 2017, 15), rotation=45)
plt.tight_layout()
plt.show()


# M2: Group by Year and Month and calculate the average for GCAG and GISTEMP
# Filter the data to get left join for GCAG
GCAG_Left = filter_GCAG

# Group by Year and Month and calculate the average monthly mean for GCAG
GCAG_All = filter_GCAG.groupby(['Year', 'Month'])['Mean'].mean().reset_index()
GCAG_All['Avg_GCAG'] = GCAG_All.groupby('Year')['Mean'].transform('mean')

# Merge the grouped data with original filter_GCAG to keep original data
GCAG_All = pd.merge(GCAG_All, GCAG_Left[['Year', 'Date', 'Mean']], on='Year', how='inner')

# Inspect the result
print(GCAG_All.head())

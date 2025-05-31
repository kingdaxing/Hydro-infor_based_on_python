
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # For proper datetime formatting


# Sample DataFrame creation (you can replace this with your actual data loading process)
# Gobal_temp = pd.read_csv("global_monthly_mean_temperature_anomaly.csv")

# Example DataFrame setup (just for the sake of illustration)
# Gobal_temp = pd.read_csv("path_to_your_file.csv")
# Assuming 'Date' column is in the format 'YYYY-MM' or 'YYYY-MM-DD'
Gobal_temp = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/global_monthly_mean_temperature_anomaly.csv")

# Ensure 'Date' is in datetime format
Gobal_temp['Date'] = pd.to_datetime(Gobal_temp['Date'])

# Subset GCAG and GIS
GCAG = Gobal_temp[Gobal_temp['Source'] == 'GCAG']
GIS = Gobal_temp[Gobal_temp['Source'] == 'GISTEMP']

# Sort by 'Date'
GCAG_new = GCAG.sort_values(by='Date')
GIS_new = GIS.sort_values(by='Date')

# Extract January data
GCAG_Jan_data = GCAG_new[GCAG_new['Date'].dt.month == 1]
GIS_Jan_data = GIS_new[GIS_new['Date'].dt.month == 1]

# Combine data for plotting
GCAG_GIS_Jan_data = pd.concat([GCAG_Jan_data, GIS_Jan_data])


sns.lineplot(data=GCAG_GIS_Jan_data, x='Date', y='Mean', hue='Source')
plt.title('Temperature anomaly in Jan: GCAG and GISTEMP')
plt.xlabel('Time')
plt.ylabel('Monthly mean anomaly Temp (°C)')
plt.xticks(rotation=45)
plt.show()

# 2.2 Annual Mean Temperature
# Remove data from 2016 (for both GCAG and GIS)
filter_GCAG = GCAG_new[GCAG_new['Source'] == 'GCAG'].copy()  # Make an explicit copy
filter_GIS = GIS_new[GIS_new['Source'] == 'GISTEMP'].copy()  # Make an explicit copy

# Now use .loc[] to safely assign values to 'Year' and 'Month'
filter_GCAG.loc[:, 'Year'] = filter_GCAG['Date'].dt.year
filter_GCAG.loc[:, 'Month'] = filter_GCAG['Date'].dt.month

filter_GIS.loc[:, 'Year'] = filter_GIS['Date'].dt.year
filter_GIS.loc[:, 'Month'] = filter_GIS['Date'].dt.month

# Continue with the rest of the operations
# For example: Compute annual mean temperature (as you did before)
annual_mean_GCAG = filter_GCAG.groupby('Year')['Mean'].mean().reset_index()
annual_mean_GCAG.rename(columns={'Mean': 'Mean_GCAG'}, inplace=True)

annual_mean_GIS = filter_GIS.groupby('Year')['Mean'].mean().reset_index()
annual_mean_GIS.rename(columns={'Mean': 'Mean_GIS'}, inplace=True)

# Merge the annual means
annual_mean = pd.merge(annual_mean_GCAG, annual_mean_GIS, on='Year')

# Plot the annual mean temperature
sns.lineplot(data=annual_mean, x='Year', y='Mean_GCAG', label='GCAG', color='blue')
sns.lineplot(data=annual_mean, x='Year', y='Mean_GIS', label='GIS', color='red')
plt.title('Annual Mean Temperature: GCAG and GISTEMP')
plt.xlabel('Year')
plt.ylabel('Annual Mean Temperature (°C)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 2.3 Compute Mean using `dplyr` equivalent in Pandas
GCAG_Left = filter_GCAG

# Group by Year and Month, then calculate mean for GCAG
GCAG_All = filter_GCAG.groupby(['Year', 'Month'])['Mean'].mean().reset_index()

# Then, group by Year to get the yearly average of GCAG
GCAG_All = GCAG_All.groupby('Year')['Mean'].mean().reset_index()
GCAG_All.rename(columns={'Mean': 'Avg_GCAG'}, inplace=True)

# Now join this with the original GCAG data for further analysis
GCAG_All = pd.merge(GCAG_All, GCAG_Left, on='Year', how='inner')
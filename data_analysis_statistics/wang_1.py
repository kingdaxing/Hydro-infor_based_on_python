import pandas as pd
from scipy.stats import kurtosis, skew


#input data
my_data = pd.read_csv("D:/PXX-NUS/00-Study/Data Sets/Sediment.csv")
print(my_data.info())

#1.2
my_data_new = my_data.iloc[10:49,:]
my_data_new.to_csv("my_data_new.txt", index=False)
#1.3
percentage_1=(my_data_new['theta']>1.5).mean()
print(f"Alternative Percentage: {percentage_1}")
#2.1
#The skipna=True parameter ensures that any NaN (missing) values are ignored when calculating the mean,MEDIAN, MODE.
column_means = my_data.mean(skipna=True)
print(f"Column means:n\{column_means}") #\n means each outcome start a newline

column_median = my_data.median(skipna=True)
print(f"Column median:n\{column_median}") #\n means each outcome start a newline

theta_mode = my_data['theta'].mode()[0]
print(f"theta mode: {theta_mode}")
#.mode function donot have a skipna/
# If there are multiple modes ,it will return all of them.
#[0] accesses the first mode value, assuming there is at least one mode.
#['theta'] use ['name of column'] format to express the whole column of my_data

#2.2
# Variance , standard deviation,Skewness and Kurtosis
column_var = my_data.var(skipna=True)
column_sds = my_data.std(skipna=True)
column_skew = my_data.skew(skipna=True)
column_kurt = my_data.apply(lambda x: kurtosis(x, nan_policy='omit'))
#.apply() FUNCTION applies the function to each column individually.
#计算每一列的峰度（kurtosis），即数据分布的陡峭程度。/峰度高表示数据分布比正态分布更尖锐，峰度低表示数据分布更平坦。
#在这里，给定的函数是一个 lambda（匿名函数），即 x 是 my_data 的一列数据。
#lambda x: kurtosis(x, nan_policy='omit'),结果是一个 pandas Series，其中每一列对应一个峰度值。
#kurtosis(x, nan_policy='omit') 的含义：计算 x 的 峰度（kurtosis）。/nan_policy='omit' 表示忽略缺失值。
print(f"Column variances:\n{column_var}")
print(f"Column standard deviations:\n{column_sds}")
print(f"Column skewness:\n{column_skew}")
print(f"Column kurtosis:\n{column_kurt}")
#这段代码展示了如何通过 pandas 提供的内置函数，计算和输出数据集中的一些基本统计指标（均值、中位数、众数、方差、标准差、偏度、峰度等）。这些统计量有助于我们理解数据的分布情况、波动性和集中趋势等特征。
my_data_properties=my_data.describe()
#pandas .describe() - provides properties of counts, mean, std, min, max, etc.
print(my_data_properties)

2.3
2.3: Handling missing data
my_data2 = my_data.copy()
my_data2.iloc[[0, 4, 14]] = np.nan  # Introducing NaNs to simulate missing data
# if in 1,5,15 columns in stead of row: my_data2.iloc[:, [0, 4, 14]] = np.nan
# =np.nan donot means zero but means null (赋值为空)

# Recalculate with NaNs ignored
column_means2 = my_data2.mean(skipna=True)
column_sds2 = my_data2.std(skipna=True)
print(f"New means (with missing data): {column_means2}")
print(f"New standard deviations (with missing data): {column_sds2}")
#print(f"文字解释 : {variable 的名字}")，格式中冒号，中括号是重点


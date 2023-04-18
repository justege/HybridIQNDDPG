import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = [
    "runs/test/PriceAndWeightDistribution_best_1.csv",
    "runs/test/PriceAndWeightDistribution_test_450_0.5.csv",
    "runs/test/PriceAndWeightDistribution_test_390_0.75.csv",
    "runs/test/PriceAndWeightDistribution_test_480_0.375.csv",
]

preprocessed_path = "Data/0001_test.csv"
data = pd.read_csv(preprocessed_path, index_col=0)
unique_trade_date = data[(data.datadate > 20171001)].datadate.unique()
# print(unique_trade_date)

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate','tic'], ignore_index=True)
    # data  = data[final_columns]
    #data.index = data.datadate.factorize()[0]
    return data

train = data_split(data, start=20210603, end=20220101)
timeframe = pd.to_datetime(train[train.tic == 'TSLA']['datadate'], format='%Y%m%d', errors='ignore')

df = pd.DataFrame()
data = pd.read_csv(files[0])
data = np.array(data)

# extract the weights and time
weights = np.array(data)[:, 1:5]
prices = np.array(data)[:, 5:9]

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 11))


ax1.stackplot(timeframe, weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)

ax1.legend(loc='upper left')
ax1.set_ylim(0, 1)
ax1.set_ylabel('Weight')

data = pd.read_csv(files[1])
weights = np.array(data)[:, 1:5]
prices = np.array(data)[:, 5:9]

date_range = timeframe

ax2.stackplot(date_range, weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)

data = pd.read_csv(files[2])
weights = np.array(data)[:, 1:5]
prices = np.array(data)[:, 5:9]

ax3.stackplot(date_range, weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)

data = pd.read_csv(files[3])
weights = np.array(data)[:, 1:5]
prices = np.array(data)[:, 5:9]

ax4.stackplot(date_range, weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)

ax5.plot(date_range, prices[:, 1], 'b-', label='AAPL')
ax5.plot(date_range, prices[:, 2], 'y-', label='CL=F')
ax5.plot(date_range, prices[:, 3], 'g-', label='TSLA')

ax1.set_title(r'$\eta=0.375$')
ax2.set_title(r'$\eta=0.5$')
ax3.set_title(r'$\eta=0.75$')
ax4.set_title(r'$\eta=1$')
ax5.set_title('Prices')
min_date = timeframe.min()
max_date = timeframe.max()

ax1.set_xlim(min_date, max_date)
ax2.set_xlim(min_date, max_date)
ax3.set_xlim(min_date, max_date)
ax4.set_xlim(min_date, max_date)
ax5.set_xlim(min_date, max_date)


plt.subplots_adjust(hspace=0.6)

# show the plot
plt.legend()
plt.show()



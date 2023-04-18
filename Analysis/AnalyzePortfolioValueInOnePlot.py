import pandas as pd
import matplotlib.pyplot as plt

csv_files = [
    "runs/test/PortfolioValueAndEqualWeight_best_1.csv",
    "runs/test/PortfolioValueAndEqualWeight_best_0.5.csv",
    "runs/test/PortfolioValueAndEqualWeight_best_0.75.csv",
    "runs/test/PortfolioValueAndEqualWeight_best_0.375.csv",
]

csv_files_labels = [
    r'$\eta=1$',
    r'$\eta=0.5$',
    r'$\eta=0.75$',
    r'$\eta=0.375$',
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

train = data_split(data, start=20210601, end=20220101)
timeframe = pd.to_datetime(train[train.tic == 'TSLA']['datadate'], format='%Y%m%d', errors='ignore')

# Loop through the CSV files and plot the second column of each file
for file, label in zip(csv_files, csv_files_labels):
    data = pd.read_csv(file, header=None)
    plt.plot(timeframe, data[1], label=label)

# Customize the plot
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Value of Different Strategies")
plt.legend()

# Display the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

preprocessed_path = "/Users/egemenokur/PycharmProjects/RL4Trading/DataSets/Second_Close_Of_Stocks.csv"
data = pd.read_csv(preprocessed_path, index_col=0)
import seaborn as sns

plt.title('Correlation of Price')

sns.heatmap(data.corr(),
        xticklabels=data.columns,
        yticklabels=data.columns)

plt.show()


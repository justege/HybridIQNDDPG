import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# list of filenames
import seaborn as sns

"""
filenames = ['/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_360_0.375.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_390_1.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_420_0.875.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_450_0.75.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_480_0.5.csv',
             ] 
             
                          '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/DailyReturnDistribution_test_450_0.75.csv',
             
             ax1.set_title(r'$\eta=0.375$')
ax2.set_title(r'$\eta=0.5$')
ax3.set_title(r'$\eta=0.75$')
ax4.set_title(r'$\eta=1$')
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filenames = [
    '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/DailyReturnDistribution_test_480_1.csv',
    '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/DailyReturnDistribution_test_390_0.75.csv'
]

dfs = []

labels = [
    r'$\eta=1$',
    r'$\eta=0.75$',
]

for file in filenames:
    data = pd.read_csv(file)
    data = data[(data.iloc[:,1] > -0.05) & (data.iloc[:,1] < 0.05)]
    dfs.append(data.iloc[:, 1])

colors = ['darksalmon',   'darkviolet']

fig, ax = plt.subplots()

ax.set_title('Daily Return Distribution')
ax.set_ylabel('Frequency')
ax.set_xlabel('Daily Return')

for i, column in enumerate(dfs):
    sns.histplot(column, bins=50, color=colors[i], alpha=0.5, label=labels[i], element='step')
    sns.kdeplot(column, color=colors[i], linewidth=2)
    plt.axvline(x=column.mean(), color=colors[i], linestyle='--', label=f'{labels[i]} mean')

plt.legend()
plt.show()

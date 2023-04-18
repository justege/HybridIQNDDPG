import matplotlib.pyplot as plt
import pandas as pd

# read the CSV file
data = pd.read_csv('/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/Results_Train_longrun500_co.csv')

# extract the first column
x = data.iloc[:, 1]


# plot the data
plt.plot(x)
plt.ylabel('Portfolio Value')
plt.xlabel('Episode')

# show the plot
plt.show()

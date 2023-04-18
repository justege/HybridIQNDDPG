import pandas as pd
import matplotlib.pyplot as plt

preprocessed_path = "/Users/egemenokur/PycharmProjects/RL4Trading/DataSets/Second_Close_Of_Stocks.csv"

# Read the data into a pandas DataFrame
df = pd.read_csv(preprocessed_path, parse_dates=['Date'], index_col='Date')

# Plot the data
plt.plot(df.index, df['AAPL'], label='AAPL')
plt.plot(df.index, df['TSLA'], label='TSLA')
plt.plot(df.index, df['CL=F'], label='CL=F')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
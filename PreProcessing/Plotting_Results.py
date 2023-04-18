
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

preprocessed_path = "/Users/egemenokur/PycharmProjects/RL4Trading/DataSets/Thesis Master Doc - Results_Oil.csv"

data = pd.read_csv(preprocessed_path)

print(data)
#train[train.index % 3 == 0]  # Ex
Sharpe06 = data['A-0.6']
Sharpe08 = data['A-0.8']
Sharpe1 =data['A-1']
frame =data['Frame']


fig, ax1 = plt.subplots()
# Define position of 1st subplot

# Set the title and axis labels
plt.title('Sharpe Ratio Evolution')
plt.xlabel('Frames')


lns1 = ax1.plot(frame,Sharpe06,label='0.6',color='r')
lns2 = ax1.plot(frame,Sharpe08,label='0.8',color='b')
lns3 = ax1.plot(frame,Sharpe1,label='1',color='g')



lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.spines['right'].set_color('yellow')
plt.grid()
plt.show()


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

preprocessed_path = "/Users/egemenokur/PycharmProjects/RL4Trading/DataSets/Processed_AAPLTSLACL=F.csv"

data = pd.read_csv(preprocessed_path, index_col=0)

unique_trade_date = data[(data.datadate > 20110101)].datadate.unique()
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

train = data_split(data, start=20110101, end=20230101)


timeframe = pd.to_datetime(train[train.index % 3 == 1]['tic'], format='%Y%m%d', errors='ignore')

timeframe.to_csv('findIndexOfERRORS.csv')
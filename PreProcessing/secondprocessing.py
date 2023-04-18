import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

### Create the Stacked LSTM model


from stockstats import StockDataFrame as Sdf

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    #df = df.sort_values(['tic'], ignore_index=True)
    print(df)
    stock = Sdf.retype(df)
    print(stock)

    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    # temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)

    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df


ADDITIONAL = 1
def preprocess_data(df):
    """data preprocessing pipeline"""


    # get data after 2009
    df['datadate'] = ((df['datadate_full'].dt.year * (10000 * ADDITIONAL)) + (df['datadate_full'].dt.month * (100 * ADDITIONAL)) + ((df['datadate_full'].dt.day) * ADDITIONAL))

    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    # add technical indicators using stockstats
    #df = df.sort_values(['datadate', 'tic'], ignore_index=True)
    df_final=add_technical_indicator(df)

    # fill the missing values at the beginning
    df_final = df_final.sort_values(['datadate', 'tic'], ignore_index=True)
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)

    return _data


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]



    # data  = data[final_columns]
    data.index = data.datadate.factorize()[0]


    return data


df = yf.download(['AAPL','TSLA','CL=F'], start = '2016-01-01', end = '2022-12-01')

#print(df['Close']['AAPL'])
#print(df['Close']['AAPL'])
#print(df['Close']['CL=F'])

df['AAPL'] = 'AAPL'
df['TSLA'] = 'TSLA'
df['CL=F'] = 'CL=F'

print(df)

df1_close = pd.DataFrame(list(zip(df['Close']['AAPL'], df['Close']['TSLA'], df['Close']['CL=F'])), index= df.index )

df1_close_to_csv = df1_close
df1_close_to_csv.columns =['AAPL','TSLA','CL=F']
df1_close_to_csv.to_csv("Second_Close_Of_Stocks.csv")
df1_volume = pd.DataFrame(list(zip(df['Volume']['AAPL'], df['Volume']['TSLA'], df['Volume']['CL=F'])))

df1_Vol_to_csv = df1_volume
df1_Vol_to_csv.columns =['AAPL','TSLA','CL=F']
df1_Vol_to_csv.to_csv("Second_Vol_Of_Stocks.csv")
df1_low = pd.DataFrame(list(zip(df['Low']['AAPL'], df['Low']['TSLA'], df['Low']['CL=F'])))
df1_high = pd.DataFrame(list(zip(df['High']['AAPL'],  df['High']['TSLA'], df['High']['CL=F'])))
df1_open = pd.DataFrame(list(zip(df['Open']['AAPL'],  df['Open']['TSLA'], df['Open']['CL=F'])))

df1_close = pd.DataFrame(df1_close.stack().squeeze()).reset_index()
df1_volume = pd.DataFrame(df1_volume.stack().squeeze()).reset_index().drop(columns=['level_1','level_0'])
df1_low = pd.DataFrame(df1_low.stack().squeeze()).reset_index().drop(columns=['level_1','level_0'])
df1_high = pd.DataFrame(df1_high.stack().squeeze()).reset_index().drop(columns=['level_1','level_0'])
df1_open = pd.DataFrame(df1_open.stack().squeeze()).reset_index().drop(columns=['level_1','level_0'])




df2 = pd.concat([df1_close, df1_volume, df1_low, df1_high, df1_open], axis=1)
#df2 = pd.DataFrame(list(zip(df1_close, df1_volume, df1_low, df1_high, df1_open)))

df2.columns =['datadate_full', 'tic', 'Close', 'Volume','Low','High','Open']

print(df2)

df2.loc[df2.tic == 0, 'tic'] = 'AAPL'
df2.loc[df2.tic == 1, 'tic'] = 'TSLA'
df2.loc[df2.tic == 2, 'tic'] = 'CL=F'
print(df2)
#df.groupby(by="Date", dropna=False)
#print(df2.to_string())




df2.datadate_full = pd.to_datetime(df2['datadate_full'], format='%Y%m%d', errors='ignore')



data = preprocess_data(df2)
data.to_csv('Processed_AAPLTSLACL=F.csv')



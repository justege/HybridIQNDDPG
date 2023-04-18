import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import csv
import matplotlib as plt
import pandas as pd
import yfinance as yf
import nasdaqdatalink
import datetime
import investpy
from iexfinance.refdata import get_symbols
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
import requests


import pandas_datareader as pdr


#Data viz
import plotly.graph_objs as go

Symbols = ['AAPL']

def AlpacaData(Symbols):
    BASE_URL = "https://paper-api.alpaca.markets"
    KEY_ID = ""
    # Instantiate REST API Connection
    SECRET_KEY = ""
    api = tradeapi.REST(key_id=KEY_ID,secret_key=SECRET_KEY,base_url="https://paper-api.alpaca.markets")
    barTimeframe = "1D"  # 1Min, 5Min, 15Min, 1H, 1D

    # Fetch Account
    account = api.get_account()
    # Print Account Details
    print(account.id, account.equity, account.status)

    iteratorPos = 0  # Tracks position in list of symbols to download

    for Symbol in Symbols:
        # Fetch Apple data from last 100 days
        Alpaca_DataFrame = api.get_bars(Symbol, barTimeframe, start="2010-01-01", adjustment='raw').df
        Alpaca_DataFrame['Symbol'] = Symbol
        # Preview Data
        #Alpaca_DataFrame.to_csv('Alpaca_Data.csv', mode='a', index=False, header=False) #Appending mode
        Alpaca_DataFrame.to_csv('DataSets/CSVs/01_Alpaca_Data.csv')

def YfinanceData(Symbols):
    # Interval required 5 minutes
    start = datetime.datetime(2015, 12, 1)
    for Symbol in Symbols:
        Yfinance_DataFrame = yf.download(tickers=Symbol, interval='1d', start= start)
        Yfinance_DataFrame['Symbol'] = Symbol
        Yfinance_DataFrame.to_csv('DataSets/CSVs/02_Yfinance_Data.csv')


def QuandlData(Symbols): # Or NasdaqData since it is acquired by nasdaq

    #Unfortunately, it has it's own naming convention for symbols :(
    Symbols = ['WIKI/AAPL.4']
    for Symbol in Symbols:
        Nasdaq_Dataframe = nasdaqdatalink.get(Symbol, start_date="2001-12-31", end_date="2021-12-31")
        Nasdaq_Dataframe['Symbol'] = Symbol
        Nasdaq_Dataframe.to_csv('DataSets/CSVs/03_Nasdaq_Data.csv')


        # df.index = mdates.date2num(df.index)
        # data = df.reset_index().values # Convert dataframe into 2-D list


def InvestingData(Symbols):

    for Symbol in Symbols:
        Investing_Dataframe = investpy.get_stock_historical_data(stock=Symbol,
                                                country='United States',
                                                from_date='01/01/2018',
                                                to_date='01/01/2020')
        Investing_Dataframe['Symbol'] = Symbol
        Investing_Dataframe.to_csv('DataSets/CSVs/04_Investing_Data.csv')


def IEXCloudData(Symbols):

    for Symbol in Symbols:
        IEXCloud_Dataframe = get_historical_data(Symbol, output_format='pandas',token="",
                                                 start='01/01/2007',
                                                 end='23/06/2022'
                                                 )
        IEXCloud_Dataframe.to_csv('DataSets/CSVs/05_IEXCloud_Data.csv')


def MarketStackData(Symbols):
    r = requests.get('http://api.marketstack.com/v1/tickers/AAPL/intraday?access_key=&date_from=2015-01-01&date_to=2022-06-23')
    x = r.json()
    data = r.json()['data']

    MarketStack_Dataframe = pd.DataFrame.from_dict(data['intraday'])
    MarketStack_Dataframe.to_csv('DataSets/CSVs/06_MarketStackData_Data.csv')


#AlpacaData(Symbols)
#YfinanceData(Symbols)
#QuandlData(Symbols)
#InvestingData(Symbols)
MarketStackData(Symbols)







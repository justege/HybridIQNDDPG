import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = [
    'runs/test/PriceAndWeightDistribution_best_0.375.csv',
'runs/test/PriceAndWeightDistribution_best_0.5.csv',
'runs/test/PriceAndWeightDistribution_best_0.75.csv',
'runs/test/PriceAndWeightDistribution_best_1.csv']


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

train = data_split(data, start=20200304, end=20210301)

# Define position of 1st subplot


for indx, Stock_Number in enumerate(['AAPL', 'CL=F', 'TSLA']):
    fig = plt.figure(figsize=(10, 7))
    #train[train.index % 3 == 0]  # Ex
    Stock_Close = train[train.tic  == Stock_Number]['adjcp']
    Stock_Volume = train[train.tic == Stock_Number]['volume']
    Stock_MACD = train[train.tic == Stock_Number]['macd']
    Stock_rsi = train[train.tic == Stock_Number]['rsi']
    Stock_cci = train[train.tic == Stock_Number]['cci']
    Stock_adx = train[train.tic == Stock_Number]['adx']
    timeframe = pd.to_datetime(train[train.tic == Stock_Number]['datadate'], format='%Y%m%d', errors='ignore')


    time = timeframe.reset_index()

    # Add a legend to the axis
    # Define position of 2nd subplot
    bx = fig.add_subplot(6, 1, 2)


    # Set the title and axis labels

    df = pd.DataFrame()
    data = pd.read_csv(files[3])
    data = np.array(data)

    weights = np.array(data)[:, 1:5]

    Stock_rsi_reset = Stock_rsi.reset_index()
    prev_w = 0
    boughtAtSignalBelow40List = []
    soldAtSignalBelow40List = []
    boughtAtSignalBelow40IndexList = []
    soldAtSignalBelow40IndexList = []

    boughtAtSignalAbove60List = []
    soldAtSignalAbove60List = []
    boughtAtSignalAbove60IndexList = []
    soldAtSignalAbove60IndexList = []

    boughtAtSignalBetween40And60List = []
    soldAtSignalBetween40And60List = []
    boughtAtSignalBetween40And60IndexList = []
    soldAtSignalBetween40And60IndexList = []


    for ind, w in enumerate(weights[:,indx]):
        if ((w > prev_w) and (Stock_rsi_reset.rsi[ind]) > 60) :
            boughtAtSignalAbove60List.append(Stock_rsi_reset.rsi[ind])
            boughtAtSignalAbove60IndexList.append(ind)
        elif ((w < prev_w) and (Stock_rsi_reset.rsi[ind]) > 60) :
            soldAtSignalAbove60List.append(Stock_rsi_reset.rsi[ind])
            soldAtSignalAbove60IndexList.append(ind)
        if ((w > prev_w) and (Stock_rsi_reset.rsi[ind]) < 40) :
            boughtAtSignalBelow40List.append(Stock_rsi_reset.rsi[ind])
            boughtAtSignalBelow40IndexList.append(ind)
        elif ((w < prev_w) and (Stock_rsi_reset.rsi[ind]) < 40) :
            soldAtSignalBelow40List.append(Stock_rsi_reset.rsi[ind])
            soldAtSignalBelow40IndexList.append(ind)
        if ((w > prev_w) and (Stock_rsi_reset.rsi[ind]) > 40 and (Stock_rsi_reset.rsi[ind]) < 60) :
            boughtAtSignalBetween40And60List.append(Stock_rsi_reset.rsi[ind])
            boughtAtSignalBetween40And60IndexList.append(ind)
        elif ((w < prev_w) and (Stock_rsi_reset.rsi[ind]) > 40 and (Stock_rsi_reset.rsi[ind]) < 60) :
            soldAtSignalBetween40And60List.append(Stock_rsi_reset.rsi[ind])
            soldAtSignalBetween40And60IndexList.append(ind)
        prev_w = w

    plt.plot([time.datadate[i] for i in boughtAtSignalAbove60IndexList], boughtAtSignalAbove60List, 'g*')
    plt.plot([time.datadate[i] for i in soldAtSignalAbove60IndexList], soldAtSignalAbove60List, 'r*')
    plt.plot([time.datadate[i] for i in boughtAtSignalBelow40IndexList], boughtAtSignalBelow40List, 'go')
    plt.plot([time.datadate[i] for i in soldAtSignalBelow40IndexList], soldAtSignalBelow40List, 'ro')
    plt.plot([time.datadate[i] for i in boughtAtSignalBetween40And60IndexList], boughtAtSignalBetween40And60List, 'g.')
    plt.plot([time.datadate[i] for i in soldAtSignalBetween40And60IndexList], soldAtSignalBetween40And60List, 'r.')

    print('Behaviour for Technical Indicator: rsi')
    print('boughtAtSignalAbove60:' + str(len(boughtAtSignalAbove60List)) + ' \n' +
            'soldAtSignalAbove60:' + str(len(soldAtSignalAbove60List)) + ' \n' +
            'boughtAtSignalBelow40:' + str(len(boughtAtSignalBelow40List)) + ' \n' +
            'soldAtSignalBelow40:' + str(len(soldAtSignalBelow40List)) + ' \n' +
            'boughtAtSignalBetween40And60:' + str(len(boughtAtSignalBetween40And60List)) + ' \n' +
            'soldAtSignalBetween40And60:' + str(len(soldAtSignalBetween40And60List)) + ' \n' )

    rsi_tradedAtSignalAbove60 =  len(boughtAtSignalAbove60List) + len(soldAtSignalAbove60List)
    rsi_tradedAtSignalBelow40 =  len(boughtAtSignalBelow40List) + len(soldAtSignalBelow40List)
    rsi_tradedAtSignalBetween40And60 = len(boughtAtSignalBetween40And60List) + len(soldAtSignalBetween40And60List)


    plt.plot(timeframe,Stock_rsi, label='rsi')
    plt.xticks([])
    plt.grid()
    plt.legend()
    plt.tight_layout()

    bx = fig.add_subplot(6, 1, 3)

    Stock_adx_reset = Stock_adx.reset_index()
    prev_w = 0

    boughtAtSignalBelow25List = []
    soldAtSignalBelow25List = []
    boughtAtSignalBelow25IndexList = []
    soldAtSignalBelow25IndexList = []

    boughtAtSignalAbove25List = []
    soldAtSignalAbove25List = []
    boughtAtSignalAbove25IndexList = []
    soldAtSignalAbove25IndexList = []

    for ind, w in enumerate(weights[:,indx]):
        if ((w > prev_w) and (Stock_adx_reset.adx[ind]) > 25) :
            boughtAtSignalAbove25List.append(Stock_adx_reset.adx[ind])
            boughtAtSignalAbove25IndexList.append(ind)
        elif ((w < prev_w) and (Stock_adx_reset.adx[ind]) > 25) :
            soldAtSignalAbove25List.append(Stock_adx_reset.adx[ind])
            soldAtSignalAbove25IndexList.append(ind)
        if ((w > prev_w) and (Stock_adx_reset.adx[ind]) < 25) :
            boughtAtSignalBelow25List.append(Stock_adx_reset.adx[ind])
            boughtAtSignalBelow25IndexList.append(ind)
        elif ((w < prev_w) and (Stock_adx_reset.adx[ind]) < 25) :
            soldAtSignalBelow25List.append(Stock_adx_reset.adx[ind])
            soldAtSignalBelow25IndexList.append(ind)
        prev_w = w

    plt.plot([time.datadate[i] for i in boughtAtSignalAbove25IndexList], boughtAtSignalAbove25List, 'g*')
    plt.plot([time.datadate[i] for i in soldAtSignalAbove25IndexList], soldAtSignalAbove25List, 'r*')
    plt.plot([time.datadate[i] for i in boughtAtSignalBelow25IndexList], boughtAtSignalBelow25List, 'g.')
    plt.plot([time.datadate[i] for i in soldAtSignalBelow25IndexList], soldAtSignalBelow25List, 'r.')

    print('Behaviour for Technical Indicator: adx')
    print('boughtAtSignalAbove25:' + str(len(boughtAtSignalAbove25List)) + ' \n' +
                  'soldAtSignalAbove25:' + str(len(soldAtSignalAbove25List)) + ' \n' +
                  'boughtAtSignalBelow25:' + str(len(boughtAtSignalBelow25List)) + ' \n' +
                  'soldAtSignalBelow25:' + str(len(soldAtSignalBelow25List)) + ' \n'
                  )

    adx_tradedAtSignalAbove25 = len(boughtAtSignalAbove25List) + len(soldAtSignalAbove25List)
    adx_tradedAtSignalBelow25 = len(boughtAtSignalBelow25List)  + len(soldAtSignalBelow25List)

    # Set the title and axis labels
    plt.plot(timeframe,Stock_adx,  label='adx')
    plt.xticks([])
    plt.grid()
    plt.legend()
    plt.tight_layout()

    #plt.plot(AAPL_DF['turbulence'], label='turbulence')

    bx = fig.add_subplot(6, 1, 4)

    Stock_macd_reset = Stock_MACD.reset_index()
    prev_w = 0
    buyList = []
    buyList_index = []
    sellList = []
    sellList_index = []
    for ind, w in enumerate(weights[:,indx]):

        if w > prev_w:
            buyList.append(Stock_macd_reset.macd[ind])
            buyList_index.append(ind)
        elif w < prev_w:
            sellList.append(Stock_macd_reset.macd[ind])
            sellList_index.append(ind)
        prev_w = w

    plt.plot([time.datadate[i] for i in buyList_index], buyList, 'g.')
    # Add red dots on sellSignals values
    plt.plot([time.datadate[i] for i in sellList_index], sellList, 'r.')
    # Add green dots on buySignals values

    plt.plot(timeframe,Stock_MACD, label='macd')
    plt.xticks([])
    plt.grid()
    plt.legend()
    plt.tight_layout()

    bx = fig.add_subplot(6, 1, 5)

    Stock_cci_reset = Stock_cci.reset_index()
    prev_w = 0

    boughtAtSignalAbove100List = []
    boughtAtSignalAbove100IndexList = []
    soldAtSignalAbove100List = []
    soldAtSignalAbove100IndexList = []

    soldAtSignalBelowMinus100List = []
    soldAtSignalBelowMinus100IndexList = []
    boughtAtSignalBelowMinus100IndexList = []
    boughtAtSignalBelowMinus100List = []

    boughtAtSignalBetweenMinus100And100List = []
    boughtAtSignalBetweenMinus100And100IndexList = []
    soldAtSignalBetweenMinus100And100List = []
    soldAtSignalBetweenMinus100And100IndexList = []


    for ind, w in enumerate(weights[:,indx]):
        if ((w > prev_w) and (Stock_cci_reset.cci[ind]) > 100) :
            boughtAtSignalAbove100List.append(Stock_cci_reset.cci[ind])
            boughtAtSignalAbove100IndexList.append(ind)
        elif ((w < prev_w) and (Stock_cci_reset.cci[ind]) > 100) :
            soldAtSignalAbove100List.append(Stock_cci_reset.cci[ind])
            soldAtSignalAbove100IndexList.append(ind)
        if ((w > prev_w) and (Stock_cci_reset.cci[ind]) < -100) :
            boughtAtSignalBelowMinus100List.append(Stock_cci_reset.cci[ind])
            boughtAtSignalBelowMinus100IndexList.append(ind)
        elif ((w < prev_w) and (Stock_cci_reset.cci[ind]) < -100) :
            soldAtSignalBelowMinus100List.append(Stock_cci_reset.cci[ind])
            soldAtSignalBelowMinus100IndexList.append(ind)
        if ((w > prev_w) and (Stock_cci_reset.cci[ind]) < 100 and (Stock_cci_reset.cci[ind]) > -100) :
            boughtAtSignalBetweenMinus100And100List.append(Stock_cci_reset.cci[ind])
            boughtAtSignalBetweenMinus100And100IndexList.append(ind)
        elif ((w < prev_w) and (Stock_cci_reset.cci[ind]) < 100 and (Stock_cci_reset.cci[ind]) > -100) :
            soldAtSignalBetweenMinus100And100List.append(Stock_cci_reset.cci[ind])
            soldAtSignalBetweenMinus100And100IndexList.append(ind)
        prev_w = w

    plt.plot([time.datadate[i] for i in boughtAtSignalAbove100IndexList], boughtAtSignalAbove100List, 'g*')
    plt.plot([time.datadate[i] for i in soldAtSignalAbove100IndexList], soldAtSignalAbove100List, 'r*')
    plt.plot([time.datadate[i] for i in boughtAtSignalBelowMinus100IndexList], boughtAtSignalBelowMinus100List, 'go')
    plt.plot([time.datadate[i] for i in soldAtSignalBelowMinus100IndexList], soldAtSignalBelowMinus100List, 'ro')
    plt.plot([time.datadate[i] for i in boughtAtSignalBetweenMinus100And100IndexList], boughtAtSignalBetweenMinus100And100List, 'g.')
    plt.plot([time.datadate[i] for i in soldAtSignalBetweenMinus100And100IndexList], soldAtSignalBetweenMinus100And100List, 'r.')


    print('Behaviour for Technical Indicator: cci')
    print('boughtAtSignalAbove100:' + str(len(boughtAtSignalAbove100List)) + ' \n' +
            'soldAtSignalAbove100:' + str(len(soldAtSignalAbove100List)) + ' \n' +
            'boughtAtSignalBelowMinus100:' + str(len(boughtAtSignalBelowMinus100List)) + ' \n' +
            'soldAtSignalBelowMinus100:' + str(len(soldAtSignalBelowMinus100List)) + ' \n' +
            'boughtAtSignalBetweenMinus100And100:' + str(len(boughtAtSignalBetweenMinus100And100List)) + ' \n' +
            'soldAtSignalBetweenMinus100And100:' + str(len(soldAtSignalBetweenMinus100And100List)) + ' \n' )

    tradedAtSignalAbove100 = len(boughtAtSignalAbove100List) + len(soldAtSignalAbove100List)
    tradedAtSignalBelowMinus100 = len(boughtAtSignalBelowMinus100List) + len(soldAtSignalBelowMinus100List)
    tradedAtSignalBetweenMinus100And100 = len(boughtAtSignalBetweenMinus100And100List) + len(soldAtSignalBetweenMinus100And100List)



    # Add green dots on buySignals values
    plt.plot(timeframe,Stock_cci,  label='cci')
    plt.xticks([])
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # extract the weights and tim
    bx = fig.add_subplot(6, 1, 1)


    plt.stackplot(np.arange(weights.shape[0]), weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3],
                  labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
                  edgecolor='black',
                  alpha=0.75)
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.title(Stock_Number+' Charts')
    ax = fig.add_subplot(6, 1, 6)

    # Set the title and axis labels


    plt.plot(timeframe,Stock_Close, label='Close price')
    plt.ylabel('Close Price')
    # Hide x axis ticks
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


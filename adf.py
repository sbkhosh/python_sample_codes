#!/usr/bin/python

#!/usr/bin/python

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import fix_yahoo_finance as yf  
import statsmodels.tsa.stattools as ts
# import pandas.tseries as ts

style.use("ggplot")

def print_ht(dataframe,flg):
    if(flg == 0):
        print(dataframe.head())
    elif(flg == 1):
        print(dataframe.tail())

def create_df(tickers):
    df_all = [yf.download(tickers[i], start, end) for i in range(len(tickers))]
    [df_all[i].to_csv(str(tickers[i]+".csv")) for i in range(len(tickers))]
    [df_all[i].reset_index(inplace=True) for i in range(len(tickers))]
    [df_all[i].set_index("Date", inplace=True) for i in range(len(tickers))]

def read_df(tickers,index):
    df = pd.read_csv(str(tickers[index]+".csv"), parse_dates=True, index_col=0)
    return(df)

def plt_df(dataframe,var):
    # df.reset_index(inplace=True)
    # df.set_index("Date", inplace=True)
    df[str(var)].plot()
    plt.show()
 
def mavg_df(dataframe,wndw):
    df = dataframe
    df[str(wndw) + 'ma'] = df['Adj Close'].rolling(window=wndw).mean()

def compare(df):
    len_df = len(df)
    op_cmp = [ len(df[df['Open']>df[x]]) for x in ["Close", "Low", "Adj Close"]  ]
    op_cmp = [ x * 100.0 /len_df  for x in op_cmp ]
    strg = [ "Open_gt_Close", "Open_gt_Low", "Open_gt_Adj Close" ]
    cmp = zip(strg,op_cmp)
    return(cmp)


start = dt.datetime(2010,1,1)
end = dt.datetime.now()

tickers = ['AAPL', 'TSLA', 'GS']
index = 2
create_dat = 0
windows = [ 10, 20, 50, 100 ]

if(create_dat == 1):
    create_df(tickers)
else:
    df = read_df(tickers,index)

def stats_prices(var):
    max_var = max(var)
    min_var = min(var)
    print(max_var,min_var)
    
print(ts.adfuller(df['Adj Close'], 1))
# stats_prices(df['Close'])




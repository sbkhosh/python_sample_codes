#!/usr/bin/python3

import crypto_empyrical
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import quantstats as qs

def get_data():
    cc_name = 'BTC-USD'
    start_date = '2018-12-01'
    end_date = '2020-12-01'
    fmt_date = '%Y-%m-%d'
    
    df = pdr.DataReader(cc_name,'yahoo',pd.to_datetime(start_date,format=fmt_date).date(),\
                        pd.to_datetime(end_date,format=fmt_date).date())
    df['ret'] = df['Close'].pct_change()
    return(df[['ret']])

def sharpe_quantstats(ret):
    print(np.round(qs.stats.sharpe(ret),2))

def sharpe_jesse(ret):
    print(np.round(crypto_empyrical.sharpe_ratio(ret) * np.sqrt(252/365),2))

if __name__ == '__main__':
    df = get_data()
    sharpe_jesse(df['ret'])
    sharpe_quantstats(df['ret'])

# jesse => 1.44
# quantstats => 1.19

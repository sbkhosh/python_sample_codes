#!/usr/bin/python3

import numpy as np
import pandas as pd
import pandas_datareader as pdr

def get_data():
    cc_name = 'BTC-USD'
    start_date = '2018-12-01'
    end_date = '2020-12-01'
    fmt_date = '%Y-%m-%d'
    
    df = pdr.DataReader(cc_name,'yahoo',pd.to_datetime(start_date,format=fmt_date).date(),\
                        pd.to_datetime(end_date,format=fmt_date).date())
    return(df)


data = get_data()
print(data.pct_change())
print(data / data.shift(1) - 1)
data.iloc[1:] = data[1:].values/ data[:-1] - 1
data.iloc[0]=np.nan
print(data)



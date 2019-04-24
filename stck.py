#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib
import xlrd
import csv
from pandas import DataFrame
import datetime
from pandas_datareader import data, wb
# import pandas.io.data
from mpl_toolkits.mplot3d import Axes3D

stx=['GOOG','TXN']
len=len(stx)
ext='.csv'

for j in range(len): 
    myf_csv=stx[j]
    data=pdr.get_data_fred(myf_csv, 
                                   start=datetime.datetime(2014, 1, 1), 
                                   end=datetime.datetime(2015, 1, 1))
    data.to_csv(myf_csv+'.csv')
    df = pd.read_csv(myf_csv+'.csv', index_col = 'Date', parse_dates=True)
    df['H-L'] = df.High - df.Low
    df['100MA'] = pd.rolling_mean(df['Close'], 100)
    df['STD'] = pd.rolling_std(df['Close'], 25, min_periods=1)


    plt.figure(figsize=(10,15))
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title(myf_csv+'- close',fontsize=16)
    df['Close'].plot()

    ax2 = plt.subplot(2, 1, 2, sharex = ax1)
    ax2.set_title(myf_csv+'- vol',fontsize=16)
    df['STD'].plot()



# df[['Close','STD']].plot()
# threedee = plt.figure().gca(projection='3d')
# threedee.scatter(df.index, df['H-L'], df['Close'])
# threedee.set_xlabel('Index')
# threedee.set_ylabel('H-L')
# threedee.set_zlabel('Close')
plt.show()

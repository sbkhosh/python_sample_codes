#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

TRADING_DAYS = 365
factor_1m = 1440
factor_5m = 1440 // 5
factor_1h = 24
window = 20
flag = 'open'

filename='/home/sbkhosh/home_dir/nwd/app_new/programming/python/python_codes/projects/quant_trading/ammer_capital/data_feeds/data/BTCUSDT-1d-data.csv'

df=pd.read_csv(filename)
df=df[['timestamp','open','high','low','close','volume']]
df.columns=['time','open','high','low','close','volume']
df.set_index('time',inplace=True)
df.index=pd.to_datetime(df.index)
df['year']=df.index.year
df['month']=df.index.month
df['day']=df.index.day

returns = np.log1p(df[flag].pct_change())
returns.fillna(0, inplace=True)
volatility = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS) * 100.0

########################################################################################################################################################################
with open('ext_data.pkl', 'rb') as handle:
    data = pickle.load(handle)
   
df_1m = data['btcusdt_1m']
df_5m = data['btcusdt_5m']
df_1h = data['btcusdt_1h']

returns_1m = np.log1p(df_1m[flag].pct_change())
returns_5m = np.log1p(df_5m[flag].pct_change())
returns_1h = np.log1p(df_1h[flag].pct_change())

returns_1m.fillna(0, inplace=True)
returns_5m.fillna(0, inplace=True)
returns_1h.fillna(0, inplace=True)

volatility_1m = returns_1m.rolling(window=window*factor_1m).std() * np.sqrt(TRADING_DAYS*factor_1m) * 100.0
volatility_5m = returns_5m.rolling(window=window*factor_5m).std() * np.sqrt(TRADING_DAYS*factor_5m) * 100.0
volatility_1h = returns_1h.rolling(window=window*factor_1h).std() * np.sqrt(TRADING_DAYS*factor_1h) * 100.0

fig = plt.figure(figsize=(32,20))
ax1 = fig.add_subplot(1, 1, 1)
volatility_1m.plot(ax=ax1,label='1m')
volatility_5m.plot(ax=ax1,label='5m')
volatility_1h.plot(ax=ax1,label='1h')
volatility.plot(ax=ax1,label='1d')

ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility (%)')
ax1.set_title('Annualized rolling {} day volatility'.format(window))
plt.legend()
plt.show()

# for el1,el2 in zip(returns_1.to_list(),returns_2.to_list()):
#     print(el1-el2)




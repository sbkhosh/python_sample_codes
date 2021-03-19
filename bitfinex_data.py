#!/usr/bin/python3

import pandas as pd
from matplotlib import pyplot as plt
import requests
import statsmodels.tsa.stattools as ts 
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# getting data from Bitfinex
def get_bitfinex_asset(asset, ts_ms_start, ts_ms_end):
    url = 'https://api.bitfinex.com/v2/candles/trade:1D:t' + asset + '/hist'
    params = { 'start': ts_ms_start, 'end': ts_ms_end, 'sort': 1}
    r = requests.get(url, params = params)
    data = r.json()
    return pd.DataFrame(data)[2]

start_date = 1514768400000 # 1 January 2018, 00:00:00
end_date = 1527811199000   # 31 May 2018, 23:59:59
assets = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XMRUSD', 'NEOUSD', 'XRPUSD', 'ZECUSD']

crypto_prices = pd.DataFrame()
for a in assets:
    print('Downloading ' + a)
    crypto_prices[a] = get_bitfinex_asset(asset = a, ts_ms_start = start_date, ts_ms_end = end_date)

crypto_prices.head()

# normalized and visualize
norm_prices = crypto_prices.divide(crypto_prices.iloc[0])
plt.figure(figsize = (15, 10))
plt.plot(norm_prices)
plt.xlabel('days')
plt.title('Performance of cryptocurrencies')
plt.legend(assets)
plt.show()

# cointegration test
for a1 in crypto_prices.columns:
    for a2 in crypto_prices.columns:
        if a1 != a2:
            test_result = ts.coint(crypto_prices[a1], crypto_prices[a2])
            if(test_result[1] < 0.05):
                print(a1 + ' and ' + a2 + ': p-value = ' + str(test_result[1]))




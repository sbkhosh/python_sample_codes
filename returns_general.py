#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('TSLA.csv')

# simple geometric returns
df = data['Close']
simple_ret = df.pct_change()
print('{} %'.format(100.0 * (1.0+simple_ret).cumprod().values[-1]-1))

# log returns
log_ret = np.log1p(simple_ret)
print('{} %'.format(100.0 * np.exp(log_ret.cumsum().values[-1])-1))


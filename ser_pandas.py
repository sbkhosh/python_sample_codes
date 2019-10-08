#!/usr/bin/python3

import pandas as pd

h = ('AA', '2012-02-01', 100, 10.2)
s = pd.Series(h)

def view_data(df):
    print(df.head())

data = {'name' : ['AA', 'IBM', 'GOOG'],
        'date' : ['2001-12-01', '2012-02-10', '2010-04-09'],
        'shares' : [100, 30, 90],
        'price' : [12.3, 10.3, 32.2]}

df = pd.DataFrame(data)
df = df.set_index(['date'])
df = df.drop(['shares','price'], axis = 1)
view_data(df)

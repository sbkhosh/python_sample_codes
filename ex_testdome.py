#!/usr/bin/python3

import pandas as pd

data = [['Ivan','1a',1],['Ana','1a',2],['Olga','1a',3],['Lida','1b',5],['Boris','1c',4]]
df = pd.DataFrame(data)
df.columns = ['name','class','grade']

dct = df.groupby('class')['grade'].median().to_dict()
res = list(map(list, dct.items()))
print(res)

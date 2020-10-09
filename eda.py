#!/usr/bin/python3

import vaex
import pandas as pd
import numpy as np

from pandas_profiling import ProfileReport

profile = ProfileReport(data, 
                        title='Pandas Profiling Report', 
                        html={'style':{'full_width':True}}) profile.to_widgets()


data['region'].value_counts().plot.barh(title='Region')
data['age'].plot.hist(title='Age distribution')

df.info(memory_usage='deep')

# conversion to hdf5 file
dv = vaex.from_csv(file_path, convert=True, chunk_size=5_000_000)
type(dv)
dv = vaex.open('big_file.csv.hdf5')

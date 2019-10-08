#!/usr/bin/python3

# pip install dateutil
# pip install bottleneck
# pip install pytz
# pip install numpy
# pip install matplotlib
# pip install pandas

# from pylab import *
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib
import xlrd
import csv
from pandas import DataFrame
import datetime
import urllib

# import pandas.io.data
# from pandas_datareader import data, wb

stx=['AAPL','GOOG']
len=len(stx)
link = 'http://www.google.com/finance/historical?output=csv&q='
ext='.csv'

for j in range(len): 
    myf=link+stx[j]
    myf_csv=stx[j].lower()+ext
    myf_csvt=stx[j].lower()+"_t"+ext
    urllib.urlretrieve(myf,myf_csv)

    ifile  = open(myf_csv, "rb")
    reader = csv.reader(ifile, delimiter=',')
    reader.next() # skip first line
    ofile  = open(myf_csvt, "wb")
    writer = csv.writer(ofile, delimiter=' ')
    for row in reader:
        writer.writerow(row)

    ifile.close()
    ofile.close()


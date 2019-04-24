#!/usr/bin/python

from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from bs4 import BeautifulSoup
import requests
import urllib2
import csv
from datetime import datetime


url = "http://www.bloomberg.com/quote/SPX:IND"
page = urllib2.urlopen(url)
soup = BeautifulSoup(page, 'html.parser')

print(page)
# name_box = soup.find('h1', attrs={'class': 'name'})
# name = name_box.text.strip()

# price_box = soup.find('div', attrs={'class':'price'})
# price = price_box.text

# with open('index.csv', 'a') as csv_file:
#  writer = csv.writer(csv_file)
#  writer.writerow([name, price, datetime.now()])


# data = r.text
# soup = BeautifulSoup(data, "lxml")

# for link in soup.find_all('a'):
#     print(link.get('href'))

#df = pd.DataFrame(data, columns=['txt']) 

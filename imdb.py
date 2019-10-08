#!/usr/bin/python3

from bs4 import BeautifulSoup
import requests
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

url = 'http://www.imdb.com/chart/top'
response = requests.get(url)
soup = BeautifulSoup(response.text,features='lxml')
tr = soup.findChildren("tr")[1:]

base_0 = filter(lambda x: x is not None, [ el.find('td', {'class': 'titleColumn'}) for el in tr ])
base_1 = filter(lambda x: x is not None, [ el.find('td', {'class': 'ratingColumn imdbRating'}) for el in tr ])

title = [ el.find('a').contents[0] for el in base_0 ]
year = [ int(str(el.find('span', {'class': 'secondaryInfo'}).contents[0]).replace('(','').replace(')','')) for el in base_0 ]
rating = [ float(str(el.find('strong').contents[0])) for el in base_1 ]

res = list(zip(year,title,rating))
df = pd.DataFrame(data = res, columns = ["year", "title", "rating"])

print(df.head())

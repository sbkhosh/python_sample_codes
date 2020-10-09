#!/usr/bin/python3

import bs4 as bs
import pandas as pd
import re
import requests

def nasdaq100_tickers(url):
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    res = table.find_all('td')

    tickers = ([re.sub(r'\W+', '', str(el).replace('td','')) for el in res[1::2]])

    names = []
    for row in res:
        name = row.find('a',href=True)
        if(name != None):
            names.append(name.text)

    assert(len(tickers)==len(names))
    df = pd.DataFrame({'names': names, 'tickers': tickers})

    # keep only the Class A shares of the ticker
    # groupby names and take 1st (<=> A shares in table)
    df = df.groupby(['names']).first()
    df.reset_index(inplace=True)
    df.set_index('tickers',inplace=True)
    return(df)


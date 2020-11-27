#!/usr/bin/python3

# import pylab
# import time
# import random
# import matplotlib.pyplot as plt

# dat=[0,1]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# Ln, = ax.plot(dat)
# ax.set_xlim([0,20])
# plt.ion()
# plt.show()    
# for i in range (18):
#     dat.append(random.uniform(0,1))
#     Ln.set_ydata(dat)
#     Ln.set_xdata(range(len(dat)))
#     plt.pause(1)

import bs4 as bs
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as pdr
import plotly.graph_objects as go
import re
import string
import time
import yaml

from datetime import datetime
from dt_help import Helper
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

import dash 
from dash.dependencies import Output, Input
import dash_core_components as dcc 
import dash_html_components as html 
import plotly 
import random 
import plotly.graph_objs as go 
from collections import deque 

def get_driver():
    opts = get_options()
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=opts)
    return(driver)

def get_options():
    prefs = {"profile.default_content_settings.popups": 0,
             "download.default_directory": os.getcwd()+str('/data_out/'),
             "excludeSwitches": ["enable-automation"],
             "useAutomationExtension": False}

    opts = Options()
    opts.add_argument('--headless')
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("prefs", prefs)
    return(opts)

X = deque(maxlen = 20) 
X.append(0) 
  
Y = deque(maxlen = 20) 
Y.append(0)
  
app = dash.Dash(__name__) 

driver_1 = get_driver()
driver_1.get("https://www.tradegate.de/orderbuch_umsaetze.php?lang=en&isin=US88160R1014")

app.layout = html.Div( 
    [ 
        dcc.Graph(id = 'live-graph', animate = True), 
        dcc.Interval( 
            id = 'graph-update', 
            interval = 1000, 
            n_intervals = 0
        ), 
    ] 
) 
  
@app.callback( 
    Output('live-graph', 'figure'), 
    [ Input('graph-update', 'n_intervals') ] 
) 
def update_graph_scatter(n):
    content_1 = driver_1.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
    raw_data_1 = content_1.get_attribute('textContent')
    driver_1.refresh()
    
    X.append(X[-1]+1) 
    Y.append(float(raw_data_1.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[1])) 
  
    data = plotly.graph_objs.Scatter( 
            x=list(X), 
            y=list(Y), 
            name='Scatter', 
            mode= 'lines+markers'
    ) 
  
    return {'data': [data], 
            'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),yaxis = dict(range = [min(Y),max(Y)]),)} 

if __name__ == '__main__':

    
    app.run_server()

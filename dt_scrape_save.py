#!/usr/bin/python3

import bs4 as bs
import dash
import dash_core_components as dcc
import dash_html_components as html
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as pdr
import plotly
import plotly.graph_objects as go
import plotly.graph_objs as go
import random
import re
import string
import time
import yaml

from collections import deque
from dash.dependencies import Output, Input
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

pd.options.mode.chained_assignment = None 

class DataScrape():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))

    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.url_link_1 = self.conf.get('url_link_1')
        self.url_link_2 = self.conf.get('url_link_2')
        self.url_link_3 = self.conf.get('url_link_3')
        self.url_link_4 = self.conf.get('url_link_4')
        self.url_link_5 = self.conf.get('url_link_5')
        self.waiting_obv = self.conf.get('waiting_obv')
        self.waiting_fx = self.conf.get('waiting_fx')
        self.last_int = self.conf.get('last_int')
        
    @Helper.timing
    def get_driver():
        opts = DataScrape.get_options()
        driver = webdriver.Chrome(ChromeDriverManager().install(),options=opts)
        return(driver)

    @Helper.timing
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

    @Helper.timing
    def get_user_agents():
        software_names = [SoftwareName.CHROME.value,SoftwareName.FIREFOX.value]
        operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   

        user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
        user_agents = user_agent_rotator.get_user_agents()
        
        user_agent = user_agent_rotator.get_random_user_agent()
        headers = {'User-Agent': user_agent}
        return(headers)

    @Helper.timing
    def get_data(self):
        driver_1 = DataScrape.get_driver()
        driver_1.get(self.url_link_1)

        driver_2 = DataScrape.get_driver()
        driver_2.get(self.url_link_2)

        driver_3 = DataScrape.get_driver()
        driver_3.get(self.url_link_3)

        driver_4 = DataScrape.get_driver()
        driver_4.get(self.url_link_4)

        driver_5 = DataScrape.get_driver()
        driver_5.get(self.url_link_5)
        
        tickers = [ el.find_element_by_xpath("//h2[starts-with(@id,'isinname')]").get_attribute('textContent') for el in [driver_1,driver_2,driver_3,driver_4,driver_5]]
        tickers = [re.findall(r"[\w']+",el)[0].lower() for el in tickers]
        
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        fx_rate = pdr.DataReader("EUR/USD","av-forex",api_key=api_key)['EUR/USD']
        bid = fx_rate.loc['Bid Price']
        ask = fx_rate.loc['Ask Price']
        self.fx = (float(bid) + float(ask)) / 2.0

        start_time = time.time()
        count = 0
        
        while(True):
            count = count + 1
            if(np.round((time.time()-start_time),0)% float(self.waiting_fx) == 0.0):
                fx_rate = pdr.DataReader("EUR/USD","av-forex",api_key=api_key)['EUR/USD']
                bid = fx_rate.loc['Bid Price']
                ask = fx_rate.loc['Ask Price']
                self.fx = np.round((float(bid) + float(ask)) / 2.0,6)
                print(self.fx)
                
            content_1 = driver_1.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
            raw_data_1 = content_1.get_attribute('textContent')
            driver_1.refresh()

            content_2 = driver_2.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
            raw_data_2 = content_2.get_attribute('textContent')
            driver_2.refresh()

            content_3 = driver_3.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
            raw_data_3 = content_3.get_attribute('textContent')
            driver_3.refresh()
            
            content_3 = driver_3.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
            raw_data_3 = content_3.get_attribute('textContent')
            driver_3.refresh()

            content_4 = driver_4.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
            raw_data_4 = content_4.get_attribute('textContent')
            driver_4.refresh()

            content_5 = driver_5.find_element_by_xpath("//tbody[starts-with(@id,'um')]")
            raw_data_5 = content_5.get_attribute('textContent')
            driver_5.refresh()

            fmt = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'

            # print(raw_data_1.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[0].split(' ')[-2])
            self.res = [ float(raw_data_1.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[1]) * self.fx, 
                         float(raw_data_1.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[0].split(' ')[-1]), 
                         float(raw_data_2.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[1]) * self.fx, 
                         float(raw_data_2.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[0].split(' ')[-1]), 
                         float(raw_data_3.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[1].replace(' ','')) * self.fx, 
                         float(raw_data_3.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[0].split(' ')[-1]),
                         float(raw_data_4.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[1].replace(' ','')) * self.fx, 
                         float(raw_data_4.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[0].split(' ')[-1]),
                         float(raw_data_5.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[1].replace(' ','')) * self.fx, 
                         float(raw_data_5.replace(u'\xa0', u' ').replace('\t','').replace('\n',' ').split('   ')[0].split(' ')[-1]) ]
            
            self.res = [ np.round(el,5) for el in self.res ]
            # print(fmt.format(self.res[0], self.res[1], self.res[2], self.res[3], self.res[4], self.res[5], self.res[6], self.res[7], self.res[8], self.res[9]))

        def get_app(self):
            X = deque(maxlen = 20) 
            X.append(0) 
  
            Y = deque(maxlen = 20) 
            Y.append(0)
  
            app = dash.Dash(__name__) 


            
            # if(np.round((time.time()-start_time),0) % float(self.waiting_obv) == 0.0):
            #     print('###############################################################')
            #     print(np.round((time.time()-start_time),0) % float(self.waiting_obv) == 0.0)
            #     print('###############################################################')
                
            #     data = pd.DataFrame(data_res)
            #     data.columns = [tickers[0],tickers[0]+'_vol',tickers[1],tickers[1]+'_vol',
            #                     tickers[2],tickers[2]+'_vol',tickers[3],tickers[3]+'_vol',
            #                     tickers[4],tickers[4]+'_vol']
            #     obv_data = []

            #     for el in tickers:
            #         Data = data[[el,el+'_vol']].tail(self.last_int)
            #         Data.columns = ['price','volume']
            #         obv = list(np.where(Data['price'] > Data['price'].shift(1), Data['volume'], np.where(Data['price'] < Data['price'].shift(1), -Data['volume'], 0)).cumsum())
            #         obv_data.append([el,obv])

            #     df = pd.DataFrame(obv_data, columns = ['ticker', 'obv'])
            #     df.set_index('ticker',inplace=True)
            #     df_lists = df[['obv']].unstack().apply(pd.Series)
            #     df_lists.plot.bar(rot=0, cmap=plt.cm.jet, fontsize=8, width=0.5, figsize=(32,20))
            #     plt.xlabel('')
            #     plt.ylabel('OBV cumulative value on the last ' + str(self.last_int) + ' seconds')
            #     plt.legend()
            #     plt.savefig(self.output_directory+'/'+str(round(time.time()-start_time))+'.pdf')


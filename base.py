#!/usr/bin/python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import urllib
import csv
import pickle
import requests
import bs4 as bs
import time
import datetime
import dateutil
import os
import glob
import lxml.html as lh
import re
from math import exp, log, sqrt
from datetime import date
from scipy.stats import norm

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

params = { 'ticker': 'AAPL',
           'day_exp': 20,
           'month_exp': 9,
           'year_exp': 2019,
           'url_sp500_etf': 'https://finance.yahoo.com/quote/SPY/options?p=SPY',
           'url_sp500_opt': 'https://finance.yahoo.com/quote/^GSPC/options?date=',
           'url_sp500_spo': 'https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC',
           'url_sp500_his': 'https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC',
           'url_rates_us': 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=2019',
           'url_bills_us': 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=billRatesYear&year=2019',
           'idx_opt': 42
}

def view_data(df,nr):
    print(df.head(nr))

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find('a', {'class': 'external text', 'rel': True}).text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    with open("sp500tickers.txt", 'w') as f:
        for item in tickers:
            f.write("%s\n" % item)        
    return(tickers)

def get_data_ticker(ticker,sdate,edate):
    ticker = str(ticker)
    try: 
        print(ticker)
        quandl.ApiConfig.api_key = "M3S6cLgQ3b_czSDmKJxD"
        df = quandl.get("WIKI/" + ticker, start_date = sdate, end_date = edate)
        write_to(df,str(ticker),"csv")
    except ValueError:
        print("Error")
        print(ticker)
    return(df)
        
def read_data(ticker,flag_set_index):
    filename = 'stock_dfs/' + str(ticker) + '.csv'
    df = pd.read_csv(filename,sep=',')
    df["Date"] = pd.to_datetime(df["Date"])
    if(flag_set_index == True):
        df.set_index("Date",inplace=True)
    return(df)

def write_to(df,name,flag):
    try:
        if(flag=="csv"):
            df.to_csv('stock_dfs/'+str(name)+".csv")
        elif(flag=="html"):
            df.to_html('stock_dfs/'+str(name)+"html")
    except:
        print("No other types supported")

def d_j(j, S, K, r, v, T):
    """
    d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)}
    """
    return (log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*T)/(v*(T**0.5))

def opt_price(S, K, r, v, T, opt):
    """
    Price of a European call/Put option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    if(opt=='C'):
        return S * norm.cdf(d_j(1, S, K, r, v, T)) - \
        K*exp(-r*T) * norm.cdf(d_j(2, S, K, r, v, T))
    elif(opt=='P'):
        return -S * norm.cdf(-d_j(1, S, K, r, v, T)) + \
            K*exp(-r*T) * norm.cdf(-d_j(2, S, K, r, v, T))

def transform_date(date):
    sdate=datetime.datetime.strptime(expiration, '%d-%m-%Y').strftime('%Y-%m-%d')
    edate=datetime.datetime.strptime(sdate, '%Y-%m-%d') + datetime.timedelta(days=1)
    return(sdate,edate)

def get_nans(df):
    nulls = df.isnull().sum()
    nulls[nulls > 0]
    print(nulls)

def get_sp500_spot(url):
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    elements = doc.xpath('//div')
    to_parse = str(elements[0].text_content())
    
    start = 'SNP Real Time Price. Currency in USD'
    end = 'At close'
    res = to_parse[to_parse.find(start)+len(start):to_parse.rfind(end)]
    res = re.split('\(|\-|\+',res)[0].replace(',','')
    return(res)

def get_sp500_hist(url):
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//tr')

    col=[]

    i=0
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        col.append((name,[]))

    for j in range(1,len(tr_elements)):
        T=tr_elements[j]
        if len(T)!=len(col):
            break
    
        i=0
        for t in T.iterchildren():
            data=t.text_content()
            if i>0:
                try:
                    data=float(data)
                except:
                    pass
            col[i][1].append(data)
            i+=1
   
    dict_all={title:column for (title,column) in col}
    df=pd.DataFrame(dict_all)
    df.rename(columns={'Close*': 'Close', 'Adj Close**': 'Adj Close'},inplace=True)

    df['Date']=df['Date'].apply(dateutil.parser.parse).apply(datetime.datetime.date)   
    df.set_index('Date',inplace=True)

    cols_tit = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    cols_typ=[np.float32]*len(cols_tit)

    for el in cols_tit:
        df[el] = [str(x).replace(',','') for x in df[el]]

    df = df.astype(dict(zip(cols_tit,cols_typ)))

    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    return(df)

def get_exp_dt_unix(y,m,d):
    os.environ['TZ'] = 'UTC'
    time.tzset()

    d = datetime.date(y,m,d)
    unixtime = int(time.mktime(d.timetuple()))

    return(unixtime)

def get_sp500_cp(url):
    exp_dt_local = datetime.date(params['year_exp'],params['month_exp'],params['day_exp'])
    exp_dt_unix = get_exp_dt_unix(params['year_exp'],params['month_exp'],params['day_exp'])
    today = date.today()
    dT = int((exp_dt_local-today).days)
    
    sp500_spot = get_sp500_spot(params['url_sp500_spo'])

    page = requests.get(url+str(exp_dt_unix))
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//tr')

    col=[]

    i=0
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        col.append((name,[]))

    for j in range(1,len(tr_elements)):
        T=tr_elements[j]
        if len(T)!=len(col):
            break
    
        i=0
        for t in T.iterchildren():
            data=t.text_content()
            if i>0:
                try:
                    data=float(data)
                except:
                    pass
            col[i][1].append(data)
            i+=1


    dict_all={title:column for (title,column) in col}
    df=pd.DataFrame(dict_all)

    idx_split = df.index[df['Contract Name'] == 'Contract Name'].tolist()[0]
    df_call = df[df.index<idx_split]
    df_put = df[df.index>idx_split]

    cols = [ el for el in df_call.columns if('Change' not in el and 'Interest' not in el \
                                            and 'Bid' not in el and 'Ask' not in el and 'Implied' not in el) ]

    df_call=df_call[cols]
    df_put=df_put[cols]

    df_call['Last Trade Date']=df_call['Last Trade Date'].apply(dateutil.parser.parse).apply(datetime.datetime.date)
    df_put['Last Trade Date']=df_put['Last Trade Date'].apply(dateutil.parser.parse).apply(datetime.datetime.date)

    df_call.replace({'-': np.NaN},inplace=True)
    df_put.replace({'-': np.NaN},inplace=True)
    
    df_call.set_index('Last Trade Date',inplace=True)
    df_put.set_index('Last Trade Date',inplace=True)

    df_call['Volume'] = [str(x).replace(',','') for x in df_call['Volume']]
    df_put['Volume'] = [str(x).replace(',','') for x in df_put['Volume']]

    df_call['Strike'] = [str(x).replace(',','') for x in df_call['Strike']]
    df_put['Strike'] = [str(x).replace(',','') for x in df_put['Strike']]

    df_call['Last Price'] = [str(x).replace(',','') for x in df_call['Last Price']]
    df_put['Last Price'] = [str(x).replace(',','') for x in df_put['Last Price']]

    cols_tit=['Strike','Last Price','Volume']
    cols_typ=[np.float32,np.float32,np.float32]

    df_call = df_call.astype(dict(zip(cols_tit,cols_typ)))
    df_put = df_put.astype(dict(zip(cols_tit,cols_typ)))

    df_call.dropna(inplace=True)
    df_put.dropna(inplace=True)

    df_call['Type'] = 'C'
    df_put['Type'] = 'P'
    
    df_call['Expiration Date'] = df_call.apply(get_exp_dt,axis=1)
    df_put['Expiration Date'] = df_put.apply(get_exp_dt,axis=1)

    df_call['SP500 Spot'] = sp500_spot 
    df_put['SP500 Spot'] = sp500_spot

    df_call['Maturity'] = dT
    df_put['Maturity'] = dT
    
    df_call.reset_index(inplace=True)
    df_put.reset_index(inplace=True)
    
    return(df_call,df_put)

def get_exp_dt(row):
    opt_type = row['Type'][0]
    if(opt_type=='C'):
        res = re.split('SPX|C',row['Contract Name'])[1]
    elif(opt_type=='P'):
        res = re.split('SPX|P',row['Contract Name'])[1]
    dt = str(int(res[:2])+2000)+str(res[2:])
    exp_dt = datetime.datetime.strptime(dt, '%Y%m%d')
    return(exp_dt)

def yield_exp():
    return(['1mo', '2mo', '3mo', '6mo', '1yr', '2yr', '3yr', '5yr', '7yr', '10yr', '20yr', '30yr'])

def get_rates(url):
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//table')

    maturities = yield_exp()
    dates=[]
    rates=[]

    i=0
    for t in tr_elements[1][1:]:
        i+=1
        name=str(t.text_content())
        raw = re.findall(r'.',name)
        res = ''.join(raw[:8])
        res = res[:6]+str(int(res[-2:])+2000)

        dates.append(datetime.datetime.strptime(res,'%m/%d/%Y'))
        rates.append(re.findall('....',''.join(raw[8:])))
    
    dict_all={'Dates': dates,
              'Rates': rates}
    df_raw=pd.DataFrame(dict_all)

    df_rates = df_raw.Rates.apply(pd.Series)
    df_rates.columns = maturities

    df_dates = df_raw.Dates.apply(pd.Series)
    df_dates.columns = ['Dates']

    df = pd.concat([df_dates, df_rates], axis=1)    
    
    df.set_index('Dates',inplace=True)

    cols_typ=[np.float32]*len(maturities)
    df = df.astype(dict(zip(maturities,cols_typ)))

    return(df)

def select_opt(df_opt,df_rate,idx):
    opt = df_opt.iloc[idx]
    ltd = opt['Last Trade Date']

    # find the closest day to Last Trade Date
    # to have the rate date and get all tenors
    df_rate.reset_index(inplace=True)
    del_dt = min(abs(pd.to_datetime(df_rate['Dates'].values)-pd.to_datetime(ltd)))
    date_select = ltd - del_dt
    df_rate.set_index(['Dates'],inplace=True)
    rs = df_rate.loc[date_select]/100.0

    opt_params = { 'S': np.float32(opt['SP500 Spot']),
                   'K': np.float32(opt['Strike']),
                   'T': np.float32(opt['Maturity']/365.0),
                   'rs': np.float32(rs),
                   'type': opt['Type'],
                   'price': np.float32(opt['Last Price'])
    }
    
    return(opt_params)

def newton_imp_vol(opt,v):
    S = opt['S']
    K = opt['K']
    r = opt['rs'][0]
    T = opt['T']
    opt_price = opt['price']
    opt_type = opt['type']
    
    d1, d2 = d_j(1, S, K, r, v, T), d_j(2, S, K, r, v, T)

    if(opt_type=='C'):
        fx = S * norm.cdf(d_j(1, S, K, r, v, T)) - K*exp(-r*T) * norm.cdf(d_j(2, S, K, r, v, T)) - opt_price
    elif(opt_type=='P'):
        fx = -S * norm.cdf(-d_j(1, S, K, r, v, T)) + K*exp(-r*T) * norm.cdf(-d_j(2, S, K, r, v, T)) - opt_price


    vega = (1.0 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(norm.cdf(d1) ** 2) * 0.5)

    tolerance = 0.00000001
    x0 = v
    xnew  = x0
    xold = x0 - 1.0
        
    while abs(xnew - xold) > tolerance:
        xold = xnew
        print(xold,xnew)
        xnew = (xnew - fx + opt_price) / vega
        
    return abs(xnew)

def vol_method_1

def sample_opt(df_opt,df_rates):
    opt = select_opt(df_opt,df_rates,params['idx_opt'])    
    # return(newton_imp_vol(opt,0.5))
    

def sample_stock():
    tickers_all = save_sp500_tickers()
    dirc = os.path.join(os.getcwd(),"stock_dfs") 
    fls_abs = glob.glob(os.path.join(dirc,str("*.csv")))
    fls_rel = [ os.path.split(el)[1] for el in fls_abs ]     
    tickers = [ el.split(".csv")[0] for el in fls_rel ]

    call = Call('GOOG', d=params['day_exp'], m=params['month_exp'], y=params['year_exp'])
    expiration = call.expiration
    stock = call.underlying.price
    strike = call.strike

    sdate,edate=transform_date(expiration)
    stock_df = get_data_ticker(params['ticker'],sdate,edate) # quandl.get("WIKI/GOOG", start_date = '20', end_date = edate)
    
        
if __name__ ==  '__main__':
    df_rates = get_rates(params['url_rates_us'])
    df_call,df_put = get_sp500_cp(params['url_sp500_opt'])

    sample_opt(df_call,df_rates)

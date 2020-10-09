#!/usr/bin/python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import email
import glob
import os
import pandas as pd
import shutil
import smtplib
import ssl
import time
import yfinance as yf

from django.http import HttpResponse
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from get_all_tickers import get_tickers as gt
from io import BytesIO
from reportlab.pdfgen import canvas

# List of the stocks we are interested in analyzing. At the time of writing
# this, it narrows the list of stocks down to 44. If you have a list of your
# own you would like to use just create a new list instead of using this, for
# example: tickers = ["FB", "AMZN", ...]
tickers = ['ABBV','ABT','BABA','BAC','CVX','HD','JNJ','JPM','KO','LLY','MA','MRK','NKE','NVS','T','XOM']
# gt.get_tickers_filtered(mktcap_min=150000, mktcap_max=10000000)

# Check that the amount of tickers isn't more than 1800
print("The amount of stocks chosen to observe: " + str(len(tickers)))

# These two lines remove the Stocks folder and then recreate it in order to remove old stocks.
shutil.rmtree("data_out")
os.mkdir("data_out")

# Holds the amount of API calls we executed
Amount_of_API_Calls = 0

# This while loop is reponsible for storing the historical data for each ticker
# in our list. Note that yahoo finance sometimes incurs json.decode errors and
# because of this we are sleeping for 2 seconds after each iteration, also if a
# call fails we are going to try to execute it again. Also, do not make more
# than 2,000 calls per hour or 48,000 calls per day or Yahoo Finance may block
# your IP. The clause "(Amount_of_API_Calls < 1800)" below will stop the loop
# from making too many calls to the yfinance API.Prepare for this loop to take
# some time. It is pausing for 2 seconds after importing each stock.
Stock_Failure = 0  # Used to make sure we don't waste too many API calls on one Stock ticker that could be having issues
Stocks_Not_Imported = 0

i=0
while (i < len(tickers)) and (Amount_of_API_Calls < 1800):
    try:
        stock = tickers[i]
        temp = yf.Ticker(str(stock))
        Hist_data = temp.history(period="max")
        Hist_data.to_csv("data_out/"+stock+".csv")
        time.sleep(2)  # Pauses the loop for two seconds so we don't cause issues with Yahoo Finance's backend operations
        Amount_of_API_Calls += 1 
        Stock_Failure = 0
        i += 1
    except ValueError:
        print("Yahoo Finance Backend Error, Attempting to Fix")  # An error occured on Yahoo Finance's backend. We will attempt to retreive the data again
        if Stock_Failure > 5:  # Move on to the next ticker if the current ticker fails more than 5 times
            i+=1
            Stocks_Not_Imported += 1
            Amount_of_API_Calls += 1
            Stock_Failure += 1
            print("The amount of stocks we successfully imported: " + str(i - Stocks_Not_Imported))

list_files = (glob.glob("data_out/*.csv"))
new_data = [] #  2D array to hold stock name and OBV score
interval = 0
while interval < len(list_files):
    Data = pd.read_csv(list_files[interval]).tail(10)  # Gets the last 10 days of trading for the current stock in iteration
    pos_move = []  # List of days that the stock price increased
    neg_move = []  # List of days that the stock price increased
    OBV_Value = 0  # Sets the initial OBV_Value to zero
    count = 0
    while (count < 10):  # looking at the last 10 trading days
        if Data.iloc[count,1] < Data.iloc[count,4]:  # True if the stock increased in price
            pos_move.append(count)  # Add the day to the pos_move list
        elif Data.iloc[count,1] > Data.iloc[count,4]:  # True if the stock decreased in price
            neg_move.append(count)  # Add the day to the neg_move list
        count += 1
    count2 = 0
    for i in pos_move:  # Adds the volumes of positive days to OBV_Value, divide by opening price to normalize across all stocks
        OBV_Value = round(OBV_Value + (Data.iloc[i,5]/Data.iloc[i,1]))
    for i in neg_move:  # Subtracts the volumes of negative days from OBV_Value, divide by opening price to normalize across all stocks
        OBV_Value = round(OBV_Value - (Data.iloc[i,5]/Data.iloc[i,1]))
    Stock_Name = ((os.path.basename(list_files[interval])).split(".csv")[0])  # Get the name of the current stock we are analyzing
    new_data.append([Stock_Name, OBV_Value])  # Add the stock name and OBV value to the new_data list
    interval += 1
df = pd.DataFrame(new_data, columns = ['Stock', 'OBV_Value'])
df["Stocks_Ranked"] = df["OBV_Value"].rank(ascending = False)
df.sort_values("OBV_Value", inplace = True, ascending = False)
df.to_csv("data_out/OBV_Ranked.csv", index = False)


Analysis = pd.read_csv("data_out/OBV_Ranked.csv")
top10 = Analysis.head(10)

subject = "report"
# body = """\
# Subject: Daily Stock Report

# Your highest ranked OBV stocks of the day:

# """ + top10.to_string(index=False) + """\

# """

sender_email = "sbkhosh@gmail.com"
receiver_email = "sbkhosh@gmail.com"
# filenm = "docs/cv_sk.pdf"
password = "ewskaukcgicoytij"
email = MIMEMultipart()
email["From"] = sender_email
email["To"] = receiver_email 
email["Subject"] = subject

body = """\
<html>
  <head></head>
  <body>
    <p>Hi!<br>
       Here is first Data Frame data:<br>
       {0}

       Regards,
    </p>
  </body>
</html>

""".format(df.to_html())

email.attach(MIMEText(body, "html"))
# attach_file = open(filenm, "rb")
# report = MIMEBase("application", "octate-stream")
# report.set_payload((attach_file).read())
# encoders.encode_base64(report)
# email.attach(report)
session = smtplib.SMTP('smtp.gmail.com:587')
session.ehlo()
session.starttls()
session.login(sender_email, password)
text = email.as_string()
session.sendmail(sender_email, receiver_email, text)
session.quit()
print('Mail Sent')
    

#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def view_data(df):
    print(df.head())

def dataframe_1():
    names = [ "Bob", "Jessica", "Mary", "John", "Mel"]
    births = [968 , 155 , 77 , 578 , 973]

    res = list(zip(names,births))
    df = pd.DataFrame(data = res, columns = ["Names", "Births"])
    df.plot()
    plt.show()

def dataframe_2():
    df = pd.DataFrame(np.random.rand(100000,5), columns = ["a","b","c","d","e"])
    view_data(df)
    print(df.describe())

def dataframe_3():
    df = pd.DataFrame(np.random.randn(100000,5), columns = ["a","b","c","d","e"])
    view_data(df)
    print(df.apply(lambda x: x.max()))

def get_data():
    df = pd.read_csv('file.csv')
    print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)])
    
get_data()


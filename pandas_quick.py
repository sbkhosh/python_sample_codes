#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataframe_1():
    names = [ "Bob", "Jessica", "Mary", "John", "Mel"]
    births = [968 , 155 , 77 , 578 , 973]

    res = list(zip(names,births))
    df = pd.DataFrame(data = res, columns = ["Names", "Births"])
    # df.columns = [''] * len(df.columns)
    # df = df.to_string(index=False)
    df.plot()
    plt.show()

def view_data(df):
    print(df.head())

def dataframe_2():
    df = pd.DataFrame(np.random.randn(3,5), columns = ["a","b","c","d","e"])
    view_data(df)
    print(df.describe())

def dataframe_3():
    df = pd.DataFrame(np.random.randn(3,5), columns = ["a","b","c","d","e"])
    print(df)    
    print("############################################################")
    print(df.apply(lambda x: x.max()))

    
dataframe_1()



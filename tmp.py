#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import urllib
import nltk
from sklearn.metrics import mean_absolute_error

# df = pd.read_excel("data.xlsx")

def view_data(df):
    print(df.head())

def get_headers(df):
    print(df.columns.values)
    
def scatter_plot(df):
    scatter_matrix(df,alpha = 0.3,figsize = (6,6),diagonal = 'kde')
    plt.show()

def plots():
    df = pd.read_csv('dudybar.txt', sep=" ", header=None)
    df.columns = ["time", "up", "down"]
    plt.scatter(x=df["up"], y=df["down"],alpha=0.3,s=30)
    plt.xlabel('up')
    plt.ylabel('down')
    plt.title('correlation')
    # df.plot(x="time",y="up",legend='')
    plt.show()
    # scatter_plot(df)

def get_freq(tokens):
    freq = nltk.FreqDist(tokens)
    return(freq)
   
    
if __name__ == '__main__':
    x = "The war owes its historical significance to multiple factors. By its end, feudal armies had been largely replaced by professional troops, and aristocratic dominance had yielded to a democratisation of the manpower and weapons of armies. Although primarily a dynastic conflict, the war gave impetus to ideas of French and English nationalism. The wider introduction of weapons and tactics supplanted the feudal armies where heavy cavalry had dominated, and artillery became important. The war precipitated the creation of the first standing armies in Western Europe since the time of the Western Roman Empire, thus helping to change their role in warfare. With respect to the belligerents, in France, civil wars, deadly epidemics, famines, and bandit free-companies of mercenaries reduced the population drastically. In England, political forces over time came to oppose the costly venture. The dissatisfaction of English nobles, resulting from the loss of their continental landholdings, as well as the general shock at losing a war in which investment had been so great, became factors leading to the Wars of the Roses"
    # urllib.urlopen('https://www.google.com/')
    xr_split = x.split()
    words = {}
    for el in xr_split:
        if el in words:
            words[el]+=1
        else:
            words[el]=1


    freq = get_freq(words)
    freq.plot(20,cumulative=False)

    # data=np.loadtxt(fname='dudybar.txt', delimiter=' ')
    # df.plot('up','down')
    # plt.show()
    
    # x = np.linspace(-np.pi,np.pi,100)
    # y = np.sin(x)
    # plt.plot(x,y,color = 'red', linestyle = 'dashed', linewidth = 1)

# def count_letters(word):
#     letters = list(word)
#     set_letters = set(letters)
#     dct_letters = {}
#     for el in set_letters:
#         dct_letters[str(el)] = letters.count(el)
#     return(dct_letters)

# word = "assfhjewppp"
# print(count_letters(word))

# class Rectangle:
#     def __init__(self,h,l):
#         self.heigth = h
#         self.length = l

#     def surface(self):
#         return(self.heigth*self.length)

#     def perimeter(self):
#         return(2 * (self.heigth + self.length))
    
# A = Rectangle(2,3)
# print(A.surface())
    

# allGuests = { 'Alice': {'apples':5, 'pretzels':12 },
#               'Bob': {'apples':6, 'pretzels':4 } }

# def total_brought(guests,itm):
#     num = 0
#     for k,v in guests.items():
#         print(k,v,v.get(itm,0))
#         num = num + v.get(itm,0)
#     return(num)

# res = total_brought(allGuests,'apples')
# print(res)

# }

# def get_paths(names,ext): 
#     cwd = os.getcwd()
#     full_paths = [os.path.join(cwd, names[x] + ext) for x in range(len(names))]
#     return(full_paths)

# def read_csvs(paths):
#     data = [pd.read_csv(paths[x]) for x in range(len(paths))]
#     return data

# def size_dfs():
#     szs = [ len(dfs[x]) for x in range(len(dfs)) ]
#     print(szs)
    
# ext = '.csv'
# names = [ 'GS', 'AAPL', 'TSLA']
# paths = get_paths(names,ext)
# dfs = read_csvs(paths)
# size_dfs()

# spam = ["apples","bananas","tofu","cats"]

# def transform(x):
#     res = []
#     for i in range(len(x)-2):
#         res.extend(x[i] + " ,")
#     res.extend(x[-2] + " and " + x[-1])
#     print(''.join(res))
    
# transform(spam)

# def sum(a,b):
#     s = a+b
#     print "Sum = %.2f" % s

# n1 = input()
# n2 = input()

# sum(n1,n2)

# list = ["a","c","h"];

# listc = ",".join(list)
# print(listc)

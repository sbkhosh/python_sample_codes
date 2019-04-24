#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
# from nltk.book import *

# txt1 = nltk.book.text1
# txt1.dispersion_plot(['Moby', 'Ahab'])
# txt1.concordance("Moby")

f_obj  = open("sample.txt", "r")
cnt = f_obj.read().split()

word_freq = {}
for tok in cnt:
    if tok in word_freq:
        word_freq[tok]+=1
    else:
        word_freq[tok]=1

f = pd.Series(word_freq)

def get_freq(tokens):
    freq = nltk.FreqDist(tokens)
    return(freq)

print(type(f))

# str = "fuck gS\n"
# splt = str.split()
# strp = str.strip()
# uppr = str.upper()
# lowr = str.lower()
# capt = str.capitalize()

# f = [ splt, strp, uppr, lowr, capt]
# f = pd.Series(f, index = ['splt', 'strp', 'uppr', 'lowr', 'capt'])
# print(f)



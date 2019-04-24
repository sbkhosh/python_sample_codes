#!/usr/bin/python

from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import time

def max_call(S,K):
    if(S>K):
        return S-K
    else:
        return 0

def max_put(S,K):
    if(S>K):
        return 0
    else:
        return K-S

    
S0, K, T, r, vol = 100, 105, 1, 0.05, 0.20

mu, sigma, Ndraws = 0, 1, 1000
z = np.random.normal(mu, sigma, Ndraws)

alpha = (r - vol * vol/2) * T
pre_norm = vol * np.sqrt(T)

t1 = time.time()
ST = S0 * np.exp(alpha + pre_norm * z)
Tvec = vectorize(max_call)
hT = Tvec(ST,K)
t2 = time.time()
diff = t2 - t1
print("Execution time = %g" % diff)

C0 = np.exp(-r * T) * np.sum(hT)/Ndraws
print("Call price  = %.5f" % C0)

# pl.plot(ST[100:200], hT[100:200], color="blue", linewidth = 1.0, linestyle = "--")
# pl.show()

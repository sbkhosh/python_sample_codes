#!/usr/bin/python

import time
import numpy as np
import matplotlib.pyplot as plt

def func_test(sample):
    t1 = time.time()
    X = range(sample)
    Y = range(sample)
    Z = []
    for i in range(len(X)):
        Z.append(X[i]+Y[i])
    return(time.time()-t1)

def func_np(sample):
    t1 = time.time()  
    X = np.arange(sample)
    Y = np.arange(sample)
    Z = X+Y
    return(time.time()-t1)

rng = [ 10000, 20000, 30000, 40000, 50000, 100000, 500000, 1000000, 5000000, 10000000 ]
ratio = []
for el in rng:
    base = func_test(el)
    opt = func_np(el)        
    ratio.append(base/opt)

plt.plot(rng,ratio)
plt.show()


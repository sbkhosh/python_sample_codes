#!/usr/bin/python3

import numpy as np
import timeit
from timeit import Timer

def test1(x,y):
    res = np.round(((1.0/np.log(2.0)) * np.log((x<<y)/x)) * np.log(x))
    res = np.exp(res)
    return(res)

x = 5
y = 3
t1 = Timer("test1(x,y)", "from __main__ import test1, x, y")

for el in list(map(int,[1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5])):
    print(el, t1.timeit(number=el))


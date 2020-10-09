#!/usr/bin/python3

import numpy as np
import timeit

def wrapper(func,*args,**kwargs):
    def wrapped():
        return(func(*args,**kwargs))
    return(wrapped)

def get_inv(num):
    reverse = 0
    while (num > 0):
        lastDigit = num % 10
        reverse = (reverse * 10) + lastDigit
        num = num // 10
    return(reverse)
        
intg, nsample = 32145678, int(1e6)
wrapped = [ wrapper(get_inv,intg) ]
res = [ timeit.timeit(el, number=nsample) for el in wrapped ]
print(res)


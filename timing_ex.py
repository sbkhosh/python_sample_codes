#!/usr/bin/python3

import numpy as np
import timeit

from timeit import Timer

def test1(lst):
    prd = np.prod(lst)
    res = [ prd/el for el in lst]
    return(res)

# if __name__ == '__main__':
arry = [1,2,3,4,5]
test1(arry)

t1 = Timer("test1(arry)", "from __main__ import test1,arry")

for el in [1000000,1500000,1750000,2000000]: # 10000,25000,50000,75000,100000,250000,500000,750000,
    print(el,t1.timeit(number=10000))

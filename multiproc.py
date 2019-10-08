#!/usr/bin/python3

from multiprocessing import Pool
import time
import numpy as np
import pandas as pd

def f(x):
    return x*x

if __name__ == '__main__':
    val = int(1e6)
    
    t1 = time.time()
    with Pool(4) as p:
        p.map(f, range(val))
    t2 = time.time()
    print(t2-t1)

    t3 = time.time()
    [ f(el) for el in range(val) ]
    t4 = time.time()
    print(t4-t3)

    t5 = time.time()
    x = pd.DataFrame(np.arange(int(1e6)),columns=['x'])
    fx = x.apply(f)
    t6 = time.time()
    print(t6-t5)

    t7 = time.time()
    fvec = np.vectorize(f)
    res = fvec(range(val))
    t8 = time.time()
    print(t8-t7)
    
    

    

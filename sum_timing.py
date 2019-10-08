#!/usr/bin/python3

import time
import matplotlib.pyplot as plt

def sum_simple(nsample,ncycle):
    tavg = 0
    for nc in range(ncycle):
        start = time.time()
        S = 0
        for el in range(nsample):
            S+=el
        end = time.time()
        tavg += end-start
    return(tavg/ncycle)

if __name__ == '__main__':
    nsample = [ 10_000, 50_000, 100_000, 300_000, 500_000, 750_000, 1_000_000 ]
    ncycle = 10

    timing = []
    [ timing.append(sum_simple(el+1,ncycle)) for el in nsample ]

    plt.plot(nsample,timing,'ro')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('number of sample')
    plt.ylabel('time (seconds)')
    plt.show()

    

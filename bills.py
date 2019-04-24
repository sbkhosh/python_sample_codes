#!/usr/bin/python

import itertools as it

def comb_100(bills):
    makes_100 = []
    for n in range(1, len(bills) + 1):
        for combination in it.combinations(bills, n):
            if sum(combination) == 100:
                makes_100.append(combination)
    return(makes_100)

bills = [20,20,20,10,10,10,10,10,5,5,5,1,1,1,1,1]
print(len(comb_100(bills)))

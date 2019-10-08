#!/usr/bin/python3

import numpy as np
import itertools
import random

def get_combinations(arr):
    combs=list(itertools.combinations(arr,2))
    return(combs)
    
def run_sum_max(lst):
    # idea is to 
    arr=np.array(lst)
    idx=np.ravel(np.where(arr>0))
    combs=get_combinations(idx)
    
    res = {combs[el]:np.sum(arr[combs[el][0]:combs[el][1]+1]) for el in range(len(combs))}
    max_sub=max(res, key=res.get)
    
    print('all array combinations = ',res)
    print('subarray with max sum = ',arr[max_sub[0]:max_sub[1]+1])
    print('max val of subarray = ',res[max_sub])
    
if __name__ == '__main__':
    lst = [-2,-3,4,-1,-2,1,5,-3] # random.sample(range(-100,100), 10)
    run_sum_max(lst)

#!/usr/bin/python3

import math
import os
import random
import re
import sys

# Complete the sockMerchant function below.
def sockMerchant(n, ar):
    tot_pairs = 0
    ar_set = set(ar)
    dict_count = {}
    for el in ar_set:
        dict_count[str(el)] = ar.count(el)//2
    return(sum(dict_count.values()))
        
n = int(input())
ar = list(map(int, input().rstrip().split()))
result = sockMerchant(n, ar)
print(result)

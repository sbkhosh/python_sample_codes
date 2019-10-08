#!/usr/bin/python3

def rec_sum(n): 
    if n == 1: 
        return 1
    else: 
        return pow(n, n) + sum(n - 1) 
  
n = 3
print(rec_sum(n)) 

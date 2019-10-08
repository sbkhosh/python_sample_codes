#!/usr/bin/python3

import math as mt 
  
# Function that returns the vector  
# containing the result for the reachability  
# of the required numbers 
def reachTheNums(nums, k, d1, d2, n): 
  
    ans = [0 for i in range(n)] 
  
    gcd = mt.gcd(d1, d2) 
  
    for i in range(n): 
        x = nums[i] - k 
  
        # If distance x is coverable 
        if (x % gcd == 0): 
            ans[i] = 1
        else: 
            ans[i] = 0
  
    for i in range(n): 
        print(ans[i], end = " ")  
  
 
# Numbers to be checked for reachability 
nums = [9, 4] 
n = len(nums) 
  
# Starting number K 
k = 8
  
# Sizes of jumps d1 and d2 
d1, d2 = 3, 2
  
reachTheNums(nums, k, d1, d2, n) 

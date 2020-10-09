#!/usr/bin/python3

def get_sum(a):      
    size = len(a)
    max_so_far = 0
    max_ending_here = 0
      
    for i in range(0, size): 
        max_ending_here = max_ending_here + a[i] 
        if max_ending_here < 0: 
            max_ending_here = 0
        elif max_so_far < max_ending_here: 
            max_so_far = max_ending_here
    return max_so_far 

if __name__ == '__main__':
    arr = [34, -50, 42, 14, -5, 86] # [-2,-3,4,-1,-2,1,5,-3]
    print(get_sum_test(arr))






































    

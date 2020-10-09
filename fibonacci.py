#!/usr/bin/python3

import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def fibo_1(n):
    fib = [0]*(n + 1)
    comp = [ n ]*(n + 1)
    fib[1] = 1
    for i in range(2,n+1):
        fib[i] = fib[i - 1] + fib[i - 2]
    # res = [abs(x1 - x2) for (x1, x2) in zip(fib, comp)]
    #     # arr = [ abs(el) for el in list(np.array(fib)-np.array(comp))]
    # return(min(res))
    return(fib[n])


def fibo_2(length):
    if(length <= 1):
        return length
    else:
        return (fibo_2(length-1) + fibo_2(length-2))

def fibo_3(n): 
    if n<0: 
        print("Incorrect input") 
    # First Fibonacci number is 0 
    elif n==1: 
        return 0
    # Second Fibonacci number is 1 
    elif n==2: 
        return 1
    else: 
        return fibo_3(n-1)+fibo_3(n-2) 
     
if __name__ == '__main__':   
    n = 100
    wrapped = [ wrapper(fibo_1, n), wrapper(fibo_1, n) ]
    res = [ timeit.timeit(el, number=1000) for el in wrapped ]
    print(res)

    

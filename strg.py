#!/usr/bin/python3

def factorial(n):
    if(n==0):
        return(1)
    return(n*factorial(n-1))

def strong_num_0(num):
    lst_num=list(str(num))
    res = sum([ factorial(int(el)) for el in lst_num ])
    return(res==num)

def strong_num_1(num):
    num_base = num
    digits=[]
    while num:
        digit = num % 10
        digits.append(digit)
        num //= 10
    digits = digits[::-1]
    res = sum([ factorial(el) for el in digits ])
    return(res==num_base)
        
if __name__ == '__main__':
    num = 145 # input('input a number: ')
    print(strong_num_1(num))
    

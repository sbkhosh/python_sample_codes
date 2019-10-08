#!/usr/bin/python3

def gcd(m,n):
    while m % n != 0:
        old_m = m
        old_n = n

        m = old_n
        n = old_m % old_n

    return(n)

print(gcd(20,10))

#!/usr/bin/python3

import math

def arb(u,d,Cu,Cd,r,n,S,K):
    # u=float(input("up = "))
    # d=float(input("down = "))
    # Cu=float(input("C_up = "))
    # Cd=float(input("C_down = "))
    # r=float(input("rate = "))
    # n=float(input("iterations = "))
    # S=float(input("stock price = "))
    # K=float(input("target price = "))

    # C_old=(1/r)*(p*Cu+(1-p)*Cd)
    # print "C_old = %16.14f \t p = %16.14f" %(C_old,p) 

    p=(r-d)/(u-d)
    C_bin=0.0
    i=0
    while i <= n:
        C_bin=C_bin+(math.factorial(n)/(math.factorial(i)*math.factorial(n-i)))*p**i*(1-p)**(n-i)*max(u**i*d**(n-i)*S-K,0)
        i+=1

    C_bin=(1/r)**n*C_bin
    print "C_bin = %16.14f" % C_bin    


for s in range(1,200,5):
    arb(1.2,0.8,2.0,0.0,1.1,s,10,10)



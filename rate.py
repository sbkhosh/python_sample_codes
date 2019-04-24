#!/usr/bin/python

import math

def eff_rate(P,r,n):
    i=0
    r_eff=0.0
    while i<=n:
        r_eff=(1.0+r/n)**n-1.0
        i+=1
        P_ow=P*(r_eff+1.0)
    print "P_ow = %16.14f" % P_ow


P_ow_cont=math.exp(0.05)*10000
print "P_ow_cont = %16.14f" % P_ow_cont

e_r=[]
e_r.append(eff_rate(10000,0.05,3))

print e_r[0]


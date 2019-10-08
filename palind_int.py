#!/usr/bin/python3

def palnd(n):
    if(str(n)==str(n)[::-1]):
        return(True)
    else:
        return(False)    

n = [22,121,10,2002,1234,12344321]
print([ palnd(el) for el in n ])

#!/usr/bin/python3

def truncate(n,decimals):
    multiplier = 10 ** decimals
    return(int(n * multiplier) / multiplier)

print(truncate(1.2346,2))

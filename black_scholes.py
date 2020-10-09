#!/usr/bin/python3

import math
import scipy.stats as stats

S0 = 105.00
K = 100.00
T = 1.
r = 0.05
vola = 0.25

def BSM_call_value(S0, K, T, r, vola):
    S0 = float(S0)
    d1 = (math.log(S0 / K) + (r + 0.5 * vola ** 2) * T) / (vola * math.sqrt(T))
    d2 = d1 - vola * math.sqrt(T)
    call_value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
    - K * math.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    return(call_value)
    
print("Value of European call option is {}".format(BSM_call_value(S0, K, T, r, vola)))

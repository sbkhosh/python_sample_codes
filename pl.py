#!/usr/bin/python3

import math
import matplotlib.pyplot as plt

rates = [ el/100.0 for el in range(1,81) ]
xr = [ math.log(1+el) for el in rates ]
yr = [ math.log(2)/el for el in xr ]

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(111, xlabel='rates',ylabel='years')
plt.plot(rates,yr)
plt.grid(True)
plt.show()

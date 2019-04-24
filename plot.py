#!/usr/bin/python

import sys
from pylab import *

t = arange(0.0, 1.0, 0.01)
s = sin(2*pi*t)
ax = subplot(111)
ax.plot(t,s)
show()

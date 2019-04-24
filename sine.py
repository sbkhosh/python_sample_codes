#!/usr/bin/python

from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s, linewidth=1.0)

xlabel('x')
ylabel('y')
title('Sine function')
grid(True)
show()

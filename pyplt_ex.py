#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import math

nstep = 256
x = np.linspace(-np.pi,np.pi,nstep)
C,S = np.cos(x), np.sin(x)

plt.plot(x,C, color = 'blue', linewidth = '2.0', linestyle = '-')
plt.plot(x,S, color = 'red', linewidth = '2.0', linestyle = '-')
plt.xlabel('x');plt.ylabel('y')
plt.title('Cos and Sin functions');plt.legend(['Cos', 'Sin'])
plt.show()

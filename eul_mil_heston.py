#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

tBegin = 0
tEnd = 2
dt = .00001

t = np.arange(tBegin, tEnd, dt)
N = t.size 
IC = 0.0001
theta = 2
mu = 1.2
sigma = 0.3

sqrtdt = np.sqrt(dt)
y1 = np.zeros(N)
y1[0] = IC

y2 = np.zeros(N)
y2[0] = IC

for i in xrange(1, N):
    y1[i] = y1[i-1] + dt*(theta*(mu-y1[i-1])) + sigma*np.sqrt(y1[i-1])*sqrtdt*np.random.normal(loc=0.0, scale=1.0)+0.5*sigma*np.sqrt(y1[i-1])*sqrtdt*np.random.normal(loc=0.0, scale=1.0)*sigma*(1.0/(2.0*np.sqrt(y1[i-1])))*sqrtdt*np.random.normal(loc=0.0, scale=1.0)
    y2[i] = y2[i-1] + dt*(theta*(mu-y2[i-1])) + sigma*np.sqrt(y2[i-1])*sqrtdt*np.random.normal(loc=0.0, scale=1.0)

ax = plt.subplot(111)
ax.plot(t, y1,'-b', label='euler')
ax.plot(t, y2,'-r', label='milstein')
ax.legend(loc='lower right')
plt.show()


#!/usr/bin/python3

import random
import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt

nsteps = int(1e5)
draws = npr.randint(0,2,size=nsteps)
steps = np.where(draws>0,1,-1)
tot_walk = steps.cumsum()
plt.plot(tot_walk,color='b',linewidth=2,linestyle='-')
plt.show()

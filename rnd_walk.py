#!/usr/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt

nsteps = 10000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk)
plt.show()

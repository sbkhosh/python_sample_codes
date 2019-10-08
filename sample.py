#!/usr/bin/python3

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

sample_size = int(1e4)
mean, var = 100, 20

rnd1 = npr.normal(mean,var,sample_size)
rnd2 = npr.standard_normal(sample_size)

fig, ((ax1, ax2)) = plt.subplots(nrows = 1, ncols = 2, figsize = (16,9))

ax1.hist(rnd1, bins = 50)
ax1.set_title('normal')
ax1.set_ylabel('frequency')
ax1.grid(True)

ax2.hist(rnd2, bins = 50)
ax2.set_title('standard normal')
ax2.set_ylabel('frequency')
ax2.grid(True)

plt.show()

nstart, nend, nsample = -np.pi, np.pi, 100
grid = np.linspace(nstart, nend, nsample)

sx = np.sin(grid)
cx = np.cos(grid)

plt.plot(grid,sx,axis=ax1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

plt.plot(grid,cx,axis=ax2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.show()



#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

nstart, nend, nsample = -np.pi, np.pi, 100
grid = np.linspace(nstart, nend, nsample)
sx = np.sin(grid)

plt.plot(grid,sx)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine function')
plt.grid(True)
plt.show()

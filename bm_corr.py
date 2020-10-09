#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

n = 1000000; dt = 0.01; rho = -1.0
W1 = np.random.normal(0, np.sqrt(dt), n)
W2 = rho * W1 + np.sqrt(1 - rho **2) * np.random.normal(0, np.sqrt(dt), n)
W1_cum = np.cumsum(W1)
W2_cum = np.cumsum(W2)
plt.plot(W1_cum)
plt.plot(W2_cum)
plt.show()

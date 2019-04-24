#!/usr/bin/python

from pylab import *
import matplotlib.pyplot as plt
import numpy
import pandas as pd

# f1 = open('dudybar.txt', 'r')
# lines = f1.readlines()
# f1.close()

data = pd.read_csv('dudybar.txt', sep=" ", header=None)
data.columns = ["time", "up", "down"]
print((np.mean(data["up"])-np.mean(data["down"]))/2.0)

# rep=4200.0
# time = []
# dudylw = []
# dudyuw = []
# cf_mean = []
# dudy_mean = []

# for line in lines:
#     p1 = line.split()
#     time.append(float(p1[0]))
#     dudylw.append(float(p1[1]))
#     dudyuw.append(-float(p1[2]))
#     dudy_mean.append((float(p1[1])-float(p1[2]))/2.0)

# print np.mean(dudy_mean)

# plt.plot(time,cf,line)
# plt.show()


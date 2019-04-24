#!/usr/bin/python

import numpy as np
data=np.loadtxt(fname='patients.csv', delimiter=',')

###################################################################################################################################
# characteristic of data
# print type(data)
# print data.shape
# data_r=data.shape[0]
# data_c=data.shape[1]

###################################################################################################################################
# # mean of each row
# data_l0=data[0,:]
# data_l1=data[1,:]
# data_l2=data[2,:]
# data_l3=data[3,:]
# data_l4=data[4,:]
# data_l5=data[5,:]

# # first way
# print data_l0.mean()
# print data_l1.mean()
# print data_l2.mean()
# print data_l3.mean()
# print data_l4.mean()
# print data_l5.mean()

# # fast way
# print data.mean(axis=1)

###################################################################################################################################
# naming of file in loop
# for x in range(0, data_r):
#     filename = 'data_l%d'%(x)
#     print repr(filename)

###################################################################################################################################
# print 'first value in data:', data[0, 1]
# print 'third value in data:', data[0, 2]
# print 'middle value in data:', data[data_r/2,data_c/2]
# print data[0:2, 0:data_r]

# small = data[:2]
# print 'small is:'
# print small

# doubledata = data * 2.0
# print data
# str='#'
# str=2*data_c*str
# print str
# print doubledata

###################################################################################################################################
# from matplotlib import pyplot
# pyplot.imshow(data)
# pyplot.show()

# ave_inflammation = data.mean(axis=0)
# pyplot.plot(ave_inflammation)
# pyplot.show()

# min_inflammation = data.min(axis=0)
# pyplot.plot(min_inflammation)
# pyplot.show()

# max_inflammation = data.max(axis=0)
# pyplot.plot(max_inflammation)
# pyplot.show()

# std_inflammation = data.std(axis=0)
# pyplot.plot(std_inflammation)
# pyplot.show()
###################################################################################################################################
from matplotlib import pyplot as plt

plt.figure(figsize=(10.0, 10.0))

plt.subplot(2, 2, 1)
plt.ylabel('average')
plt.plot(data.mean(0))

plt.subplot(2, 2, 2)
plt.ylabel('max')
plt.plot(data.max(0))

plt.subplot(2, 2, 3)
plt.ylabel('min')
plt.plot(data.min(0))

plt.subplot(2, 2, 4)
plt.ylabel('standard dev')
plt.plot(data.std(0))

plt.tight_layout()
plt.show()


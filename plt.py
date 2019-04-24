#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Generate data: for N=1e6, the triangulation hogs 1 GB of memory
N = 10000
x, y = 10 * np.random.random((2, N))
rho = np.sin(3*x) + np.cos(7*y)**3

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 300), np.linspace(y.min(), y.max(), 300)
xi, yi = np.meshgrid(xi, yi)

# Interpolate; there's also method='cubic' for 2-D data such as here
zi = scipy.interpolate.griddata((x, y), rho, (xi, yi), method='linear')

plt.imshow(zi, vmin=rho.min(), vmax=rho.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
plt.show()

########################################################################################################
# N=201
# # x = np.linspace(-np.pi, np.pi, N-1, endpoint=True)

# x = np.linspace(-np.pi, np.pi, N+1)
# y=np.sin(x)
# plt.plot(x, np.sin(x),'o',color='b',markersize=5)
# plt.xlabel('Angle [rad]')
# plt.ylabel('sin(x)')
# plt.axis('tight')
# plt.show()

########################################################################################################
# from math import *

# nbSamples = 256
# xRange = (-pi,pi)
# x,y=[],[]
# for n in xrange(nbSamples):
#     k=(n+0.5)/nbSamples
#     x.append(xRange[0]+(xRange[1]-xRange[0])*k)
#     y.append(sin(x[-1]))

# plt.plot(x,y)
# plt.show()

########################################################################################################
# x = [0,1,2,3,4]
# y = [0,1,4,9,16]
# plt.plot(x,y)
# plt.axis([0,5,0,18])
# plt.show()

########################################################################################################
# f1 = open('lx0/wrms_tot.txt', 'r')
# lines = f1.readlines()
# f1.close()
# x1 = []
# y1 = []
# for line in lines:
#     p1 = line.split()
#     x1.append(float(p1[0])*179.0)
#     y1.append(float(p1[1])*4200.0/179.0)
   
# xv1 = np.array(x1)
# yv1 = np.array(y1)

# max_rms1 = yv1.max(axis=0)
# xx1, xx2 = 0, 100
# yy1, yy2 = 0, 2.5

# plt.axis([xx1, xx2, yy1, yy2])
# p1, = plt.plot(xv1,yv1,'o',color='b',markersize=8,label="lx0")

# ########################################################################################################
# f2 = open('1p5_lx0/wrms_tot.txt', 'r')
# lines = f2.readlines()
# f2.close()
# x2 = []
# y2 = []
# for line in lines:
#     p2 = line.split()
#     x2.append(float(p2[0])*179.2)
#     y2.append(float(p2[1])*4200.0/179.2)
   
# xv2 = np.array(x2)
# yv2 = np.array(y2)

# max_rms2 = yv2.max(axis=0)
# p2, = plt.plot(xv2,yv2,'x',color='r',markersize=8,label="1p5_lx0")

# ########################################################################################################
# f3 = open('2p0_lx0/wrms_tot.txt', 'r')
# lines = f3.readlines()
# f3.close()
# x3 = []
# y3 = []
# for line in lines:
#     p3 = line.split()
#     x3.append(float(p3[0])*179.3)
#     y3.append(float(p3[1])*4200.0/179.3)
   
# xv3 = np.array(x3)
# yv3 = np.array(y3)

# p3, = plt.plot(xv3,yv3,'d',color='g',markersize=8,label="2p0_lx0")

# max_rms3 = yv3.max(axis=0)

# fig.suptitle('lin - w', fontsize=20)
# plt.legend()
# plt.show()
# fig.savefig('/home/skhosh/lin_w.pdf')

# with open("lx0/wrms_tot.txt") as f:
#     data = f.read()

# data = data.split('\n')

# x = [row.split('     ')[0] for row in data]
# y = [row.split('     ')[1] for row in data]

# fig = plt.figure()

# ax1 = fig.add_subplot(111)

# ax1.set_title("Plot title...")    
# ax1.set_xlabel('your x label..')
# ax1.set_ylabel('your y label...')

# ax1.plot(x,y, c='r', label='the data')

# leg = ax1.legend()

# plt.show()

# import numpy as np
# data=np.loadtxt(fname='lx0/wrms_tot.txt', delimiter=',')

# from matplotlib import pyplot as plt

# plt.figure(figsize=(10.0, 10.0))

# plt.subplot(2, 2, 1)
# plt.ylabel('average')
# plt.plot(data.mean(0))

# plt.subplot(2, 2, 2)
# plt.ylabel('max')
# plt.plot(data.max(0))

# plt.subplot(2, 2, 3)
# plt.ylabel('min')
# plt.plot(data.min(0))

# plt.subplot(2, 2, 4)
# plt.ylabel('standard dev')
# plt.plot(data.std(0))

# plt.tight_layout()
# plt.show()


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

#!/usr/bin/env python
# This should probably be replaced with a demo that shows all
# line and marker types in a single panel, with labels.

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import numpy as np

# t = np.arange(0.0, 1.0, 0.1)
# s = np.sin(2*np.pi*t)
# linestyles = ['_', '-', '--', ':']
# markers = []
# for m in Line2D.markers:
#     try:
#         if len(m) == 1 and m != ' ':
#             markers.append(m)
#     except TypeError:
#         pass

# styles = markers + [
#     r'$\lambda$',
#     r'$\bowtie$',
#     r'$\circlearrowleft$',
#     r'$\clubsuit$',
#     r'$\checkmark$']

# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

# plt.figure(figsize=(8,8))

# axisNum = 0
# for row in range(6):
#     for col in range(5):
#         axisNum += 1
#         ax = plt.subplot(6, 5, axisNum)
#         color = colors[axisNum % len(colors)]
#         if axisNum < len(linestyles):
#             plt.plot(t, s, linestyles[axisNum], color=color, markersize=10)
#         else:
#             style = styles[(axisNum - len(linestyles)) % len(styles)]
#             plt.plot(t, s, linestyle='None', marker=style, color=color, markersize=10)
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])

# plt.show()

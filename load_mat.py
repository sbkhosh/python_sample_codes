#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from pylab import *
from matplotlib import *

# load .mat file into dictionary x
x = scipy.io.loadmat('signal.mat')

sig_A = x['A']
print sig_A
plt.plot(sig_A)
plt.show()

# does not work even with python3
# install h5py on Ubuntu
# apt-get install python-h5py
# f = h5py.File('Desktop/signal.mat','r') 
# data = f.get('data/variable1') 
# data = np.array(data) # For converting to numpy array

#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('stinkbug.png')
lum_img = img[:,:,0]
plt.imshow(lum_img)

#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import Image

img=Image.open('Master2_new.png')    
for x in xrange(img.size[0]):
    for y in xrange(img.size[1]):
        r,g,b = img.getpixel((x,y))
        img.putpixel((x,y),(r,b,g))


img.show()







# img = mpimg.imread('Master2_new.png')
# lum_img = img[:,:,0]
# plt.imshow(lum_img)


# img=Image.new("RGB",(640,480),(0,0,255))
# for x in xrange(640):
#     for y in xrange(480):
#         img.putpixel((x,y),(x/3,(x+y)/6,y/2))

# img.show()

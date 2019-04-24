#!/usr/bin/python

from PIL import Image

im = Image.open("test.jpg")
im.save("test-600.jpg", dpi=(600,600))

# size = 7016,4961
# im = Image.open('test.jpg')
# im_resized = im.resize(size, Image.ANTIALIAS)
# im_resized.save("test_resized.jpg")


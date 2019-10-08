#!/usr/bin/python3

import time
import math

def timeit1():
    s = time.time()
    for i in xrange(750000):
        z=i**.5
    print "Took %f seconds" % (time.time() - s)

def timeit2(arg=math.sqrt):
    s = time.time()
    for i in xrange(750000):
        z=arg(i)
    print "Took %f seconds" % (time.time() - s)

timeit1()
timeit2()


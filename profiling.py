#!/usr/bin/python3
import time

# list of 500 000 000 elements
size = 5000000
print("list of {} elements  elements".format(size))
to_size = range(size)
# variable 'a' accesses one element of the list

# method 1
start = time.time()

for i in range( len(to_size) ):
    a = to_size[i]
print("method 1: {} seconds".format(time.time()-start))

# method 2
start = time.time()
for ele in to_size:
    a = ele
print("method 2: {} seconds".format(time.time()-start))

# method 3
start = time.time()
for ele in range( len(to_size) ):
    a = to_size[i]
print("method 3: {} seconds".format(time.time()-start))

# method 4
start = time.time()
for idx, ele in enumerate( to_size ):
    a = ele
print("method 4: {} seconds".format(time.time()-start))

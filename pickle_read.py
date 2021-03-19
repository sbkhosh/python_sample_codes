#!/usr/bin/python3

import pickle
import sys

filename=sys.argv[1]

with open(filename, 'rb') as handle:
    data = pickle.load(handle)

print(data)




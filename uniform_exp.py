#!/usr/bin/python3

import random

totalsteps = 0
for i in range(int(1e7)):
    summ = 0.0
    steps = 0
    while summ < 1.0:
  	summ += random.random()
  	steps += 1
  	totalsteps += steps

print(summ)        
print('avg steps: ' + str(totalsteps / i))

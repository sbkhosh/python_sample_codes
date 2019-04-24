#!/usr/bin/python
import time
import numpy as np

def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0

def powers(x):
    return x**2

vect = np.linspace(1,1e7,1e7)
hvec = np.vectorize(powers)

t1 = time.time()
res = powers(vect)
t2 = time.time()
diff = t2 - t1
print(diff)

t1 = time.time()
res = hvec(vect)
t2 = time.time()
diff = t2 - t1
print(diff)


# print heaviside(10)

# arr = [];
# for i in xrange(1,10):
#     arr.append(i)

# print arr.extend(arr)    
# print arr

student = {'name': 'John', 'age': 25, 'courses': ['maths', 'comp sci']}
# print student.get('phone', 'not found')
# student['phone'] = '555-5555'
# student.update({'name': 'Jane', 'age': 26})
# tmp = student.pop('age')
# print student.items() 

# for key,val in student.items():
#     print(key,val)

# a = [1,2,3]
# b = [1,2,3]

# a = b
# print(a is b)
# print(id(a), id(b))

# condition = not False

# if condition:
#     print("True")
# else:
#     print("False")

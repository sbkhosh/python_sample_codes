#!/usr/bin/python3

import scipy.linalg as linalg
import numpy as np

A = np.array([[2., 1., 1.],
              [1., 3., 2.],
              [1., 0., 0.]])
B = np.array([4., 5., 6.])

x_solve = np.linalg.solve(A, B )

LU = linalg.lu_factor(A)
x_lu_solve = linalg.lu_solve(LU, B)
P, L, U = linalg.lu(A) # permutation, lower and upper triangular matrix

print(A-L.dot(U))
print(x_solve)
print(x_lu_solve)

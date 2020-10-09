#!/usr/bin/python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import urllib
import csv
import pickle
import requests
import bs4 as bs
import time
import datetime
import dateutil
import os
import glob
import lxml.html as lh
import re
from math import exp, log, sqrt
from datetime import date
from scipy.stats import norm
import sys
import math
import itertools as it
from collections import Counter
from itertools import combinations
from collections import namedtuple
import numpy as np
import string

from collections import namedtuple

Dimensions = namedtuple("Dimensions", "columns rows")

def compute_product(matrixA, matrixB):
    matrixA_dim = Dimensions(len(matrixA[0]), len(matrixA))
    matrixB_dim = Dimensions(len(matrixB[0]), len(matrixB))
    print(matrixA_dim)
    print(matrixB_dim)
    product = []
    for col in range(matrixA_dim.columns):
        product.append([0 for row in range(matrixB_dim.rows)])
    for i in range(matrixA_dim.rows):
        for j in range(matrixB_dim.columns):
            product[i][j] += matrixA[i][j] * matrixB[j][j]
    return product

matrixA = [[11,  3, 10, 3], [20, 1, 0, 1]]
matrixB = [[12,  1, 10], [7, 4, 0], [4, 5, 2], [5, 2, 10]]

print("Product:", compute_product(matrixA, matrixB))


# def compute_recall(m):
#     recall = m.true_positive / (m.true_positive + m.false_positive)
#     return recall

# def compute_precision(m):
#     precision = m.true_positive / (m.true_negative + m.false_negative)
#     return precision
    
# def compute_specificity(m):
#     specificity = m.true_negative / (m.true_negative + m.false_positive)
#     return specificity

# def compute_accuracy(m):
#     accuracy = (m.true_positive + m.true_negative) / (m.false_positive + m.false_negative)
#     return accuracy

# ConfusionMatrix = namedtuple('ConfusionMatrix', 'true_positive true_negative false_positive false_negative')
# m = ConfusionMatrix(30, 60, 10, 20)

# recall = compute_recall(m)
# precision = compute_precision(m)
# specificity = compute_specificity(m)
# accuracy = compute_accuracy(m)

# print('Recall:', recall, ' Precision:', precision, ' Specificity:', specificity, ' Accuracy:', accuracy)


# stopwords = get_stop_words('en')

# def find_frequent_words(sentence):
#     sentence = sentence.translate(str.maketrans('', '', string.punctuation))
#     wordlist = sentence.split()
#     wordfreq = []
#     for w in wordlist:
#         if(w not in stopwords):
#             wordfreq.append((w,wordlist.count(w)/len(wordlist)))
#     return(list(set(wordfreq)))
    

# def count(S, m, n ): 
#     if (n == 0): 
#         return 1
  
#     if (n < 0): 
#         return 0; 
  
#     if (m <=0 and n >= 1): 
#         return 0
  
#     return count( S, m - 1, n ) + count( S, m, n-S[m-1] ); 

# amount=6
# coins=[10,5,2]
# listOfCoins=["ten", "five", "two"]
# change = []

# for coin in coins:
#     holdingAmount=amount
#     amount=amount//coin
#     change.append(amount)
#     amount=holdingAmount%coin

# for i in range(len(coins)):
#   print("There's " , change[i] ,"....",  listOfCoins[i] , "pence pieces in your change" )

# coins = [10, 5, 2]
# coinsReturned = []
# for i in coins:
#   while amount >=i:
#         coinsReturned.append(i)
#         amount = amount - i
# print(coinsReturned)

# def change(amount):
#     money = ()
#     for coin in [2,5,10]:
#         num = amount/coin
#         money += (coin,) * num
#         amount -= coin * num
#     return money

# print(change(6))

# def change(cash):
#     """Given a map from denomination to the count of bills in that
#     denomination, return the set of values that can be made using a
#     subset of the bills.

#     """
#     totals = set() # Set of possible totals 
#     cash_list = list(Counter(cash).elements()) # List of all bills
#     for r in range(len(cash_list)+1):
#         for subset in combinations(cash_list, r):
#             totals.add(sum(subset))
#     return totals


# print(change)

# def change(n, coins_available, coins_so_far):
#     if(sum(coins_so_far) == n):
#         yield(coins_so_far)
#     elif(sum(coins_so_far) > n):
#         pass
#     elif(coins_available == []):
#         pass
#     else:
#         for c in change(n, coins_available[:], coins_so_far+[coins_available[0]]):
#             yield(c)
#         for c in change(n, coins_available[1:], coins_so_far):
#             yield(c)
                        
# if __name__ == '__main__':
# 	n = 17
# 	coins = [2, 5, 10]

# 	solutions = [s for s in change(n, coins, [])]
# 	print(min(solutions, key=len))

# def change(csh):
    # coins = [10, 5, 2]
    # changedue = 33
    # coinsreturned = []
    # for i in coins:
    #     while i >= changedue:
    #         coinsreturned.append(i)
    #         changedue = changedue - i

    # print(coinsreturned)
    # bills = [2,2,2,2,2,2,2,5,5,5,5,5,5,5,10,10,10,10,10,10]
    # res = []
    # for n in range(1, len(bills) + 1):
    #     for combination in it.combinations(bills, n):
    #         if sum(combination) == csh:
    #             res.append(combination)
    # res = set(res)
    # print(res)
    # lst = list(map(len,res))

    # idx = lst.index(min(lst))
    # print(set(list(res)[idx]).pop())
    # return(idx)

    # bills = [2,2,2,2,2,2,2,5,5,5,5,5,5,5,10,10,10,10,10,10]
    # res = []
    # for n in range(1, len(bills) + 1):
    #     for combination in it.combinations(bills, n):
    #         if sum(combination) == csh:
    #             res.append(combination)
    # res = set(res)    
    # return(list(map(len,res)))
    # res = []
    # for n in range(1, len(bills) + 1):
    #     for combination in it.combinations(bills, n):
    #         if sum(combination) == csh:
    #             res.append(combination)
    # res = set(res)
    # return(res)

# cash = 33
# print(change(cash))


# def compute_multiples_sum(n):
#     # Write your code here
#     # To debug: print("Debug messages...", file=sys.stderr)
#     res = sum([i for i in range(n) if i % 3 == 0 or i % 5 == 0 or i % 7 == 0])
#     return(res)

# # Ignore and do not change the code below
# def main():
#     n = int(input())
#     with redirect_stdout(sys.stderr):
#         res = compute_multiples_sum(n)
#     print(res)

# if __name__ == "__main__":
#     main()



# def nth_most_rare(elements, n):
#     """
#     :param elements: (list) List of integers.
#     :param n: (int) The n-th element function should return.
#     :returns: (int) The n-th most rare element in the elements list.
#     """
#     d = {}
#     for v in elements:
#         if v not in d:
#             d[v] = 0
#         d[v] += 1

#     return sorted([(v, k) for k, v in d.items()], reverse=True)[n+1][1]

    # dct = {}
    # for el in elements:
    #     dct.setdefault(el,0)
    #     dct[el]+=1

    # sorted_dct = sorted(dct.items(), key=lambda kv: kv[1])
    # return(sorted_dct)

# print(nth_most_rare([5, 4, 3, 2, 1, 5, 4, 3, 2, 5, 4, 3, 5, 4, 5], 2))

# def omit_by(dct, predicate=lambda x: x!=0):
   
#     return({k:v for k,v in dct.items() if predicate(v)})

    


# strg = 'hello'
# for el in list(strg):
#     print(el)

# from typing import Dict, Tuple, Callable, Iterable

# import numpy

# def model_quadratic(model_parameters: dict):
#     """
#     This is a quadratic model with a minimum at a=0.5, b=0.75, c=0.25.
#     """
#     a = model_parameters['a']
#     b = model_parameters['b']
#     c = model_parameters['c']

#     return 1.75 + (a - 0.5) ** 2 + (b - 0.75) ** 2 + (c - 0.25) ** 2

# class Problem:
#     @staticmethod
#     def grid_search(search_space: Dict[str, Iterable],
#                     scoring_func: Callable[[Dict[str, float]], float]) -> Tuple[float, Dict[str, float]]:
#         """
#         This function accepts a search space, which is a dictionary of arrays.

#         For each key in the dictionary, the respective array holds the numbers in the search space that should be
#         tested for.

#         This function also accepts a scoring_func, which is a scoring function which will return a float score given a
#         certain set of parameters.  The set of parameters is given as a simple dictionary. As an example, see
#         model_quadratic above.

#         """
#         params_keys = search_space.keys()
#         for el in params_keys:
#             print(search)
#         return 1.75, {'a': 0.5, 'b': 0.75, 'c': 0.25}


# print(Problem.grid_search({
#     'a': numpy.arange(0.0, 1.0, 0.05),
#     'b': numpy.arange(0.0, 1.0, 0.05),
#     'c': numpy.arange(0.0, 1.0, 0.05),
# }, model_quadratic))

# class MovingTotal:
#     def __init__(self):
#         self.cnt = []        

#     def append(self, numbers):
#         """
#         :param numbers: (list) The list of numbers.
#         """
#         self.cnt.extend(numbers)        

#     def contains(self, total):
#         """
#         :param total: (int) The total to check for.
#         :returns: (bool) If MovingTotal contains the total.
#         """
#         n = len(self.cnt)//3
#         res = [ sum(self.cnt[el:el+3]) for el in list(range(n)) ]
#         print(res)
#         return(res)
    
# if __name__ == "__main__":
#     movingtotal = MovingTotal()
#     movingtotal.append([1, 2, 3])
#     # print(movingtotal.contains(6))
#     # print(movingtotal.contains(9))
#     movingtotal.append([4])
#     print(movingtotal.contains(9))

# def class_grades(students):
#     """
#     :param students: (list) Each element of the list is another list with the 
#       following elements: Student name (string), class name (string), student grade (int).
#     :returns: (list) Each element is a list with the following 
#       elements: Class name (string), median grade for students in the class (float).
#     """
#     clss = set([ el[1] for el in students ])
#     for cls in clss:
#         for i,sts in enumerate(students):
#             if(sts[i])
        
        
# students = [["Ana Stevens", "1a", 5], ["Mark Stevens", "1a", 4], ["Jon Jones", "1a", 2], ["Bob Kent", "1b", 4]]
# class_grades(students)

#!/usr/bin/python3

import numpy as np
import time

from functools import wraps
from itertools import combinations

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print('function:%r took: %6.6f sec' % (f.__name__,  end - start))
        return(result)
    return wrapper

@timing
def factor_number(n):
    res = [ x  for x in range(1,n+1) if(n%x==0) ]
    return(res)

@timing
def perfect_number(n):
    res = factor_number(n)[:-1]
    if(np.sum(res)==n):
        print('perfect number')
        
@timing
def running_sum(n):
    res = [0] * len(lst)
    res[0] = lst[0]
    for i,el in enumerate(lst):
        res[i] = res[i-1]+lst[i]
    return(res)

@timing
def replace_el_right(arr):
    out = [-1]
    greatest = 0
    for num in arr[::-1]:
        if greatest < num:
            greatest = num
        out.append(greatest)
    out.pop()
    return out[::-1]

@timing
def all_duplicates(nums):
    dct = {}
    for el in nums:
        dct.setdefault(el,0)
        dct[el]+=1
        if(dct[el]>2):
            continue
    res = [ k for k,v in dct.items() if v==2 ]
    return(res)

@timing
def two_sum(nums,target):
    # select = list(filter((target).__ge__, nums))
    complt = [target-el for el in nums]
    res = [x1 * x2 for (x1, x2) in zip(nums,complt)]
    idx_1 = list(np.diff(res)).index(0)
    idx_2 = idx_1 + 1
    return([idx_1,idx_2])

class ListNode:
    def __init__(self, val=0, next=None):
        self.dataval = val
        self.nextval = next

class SLinkedList:
    def __init__(self):
        self.headval = None

    def listprint(self):
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval
        
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    sumval = 0
    root = curr = ListNode(0)
    while l1 or l2 or sumval:
        if l1: sumval += l1.val; l1 = l1.next
        if l2: sumval += l2.val; l2 = l2.next
        curr.next = curr = ListNode(sumval % 10)
        sumval //= 10
    return root.next

if __name__ == '__main__':
    S1 = SLinkedList()
    S1.headval = ListNode(2)
    n2 = ListNode(4)
    n3 = ListNode(3)
    S1.headval.nextval = n2
    n2.nextval = n3

    S2 = SLinkedList()
    S2.headval = ListNode(5)
    p2 = ListNode(6)
    p3 = ListNode(4)
    S2.headval.nextval = p2
    p2.nextval = p3

    addTwoNumbers(S1,S2)

    

        

#!/usr/bin/python3

def fixed_point(A):
    low = 0
    high = len(A) - 1

    while(low <= high):
        mid = (low+high) // 2

        if(A[mid] < mid):
            low = mid + 1
        elif(A[mid] > mid):
            high = mid - 1
        else:
            return(A[mid])
    return(None)

def fixed_point_linear(A):
    for el in range(len(A)):
        if(A[i]==i):
            return(A[i])
    return(None)

if __name__ == '__main__':
    lst_test = [0,-5,1,4,4]
    print(fixed_point(lst_test))

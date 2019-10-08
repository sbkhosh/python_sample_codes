#!/usr/bin/python3

def pr_odd_even(lst):
    lst_sort = sorted(lst)
    felem = lst_sort[0]
    print(felem)
    evens = [ el for el in lst_sort if el%2==0 ]
    odds = [ el for el in lst_sort if el%2==1 ]

    res = []
    if(felem%2==0):
        tmp = evens + odds[::-1]
    else:
        tmp = odds + evens[::-1]

    n = len(tmp)

    i=0
    while(i<n-min(len(odds),len(evens))):
        res.append([tmp[i]] + [tmp[n-1-i]])
        i+=1

    res = list(set([ y for x in res for y in x ]))
    return(res)
    
if __name__ == '__main__':
    lst = [ 1, 3, 2, 5, 4, 7, 10 ]
    # lst = [ 9, 8, 13, 2, 19, 14 ]
    res = pr_odd_even(lst)
    print(res)
    

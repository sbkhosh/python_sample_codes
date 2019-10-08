#!/usr/bin/python3

def get_pairs(a):
    count = 0

    for i in range(len(a)):
        for j in range(len(a)):
            if(a[i]<a[j]):
                count += 1
    return(count)
        

a = [2,3,4,1]
print(get_pairs(a))

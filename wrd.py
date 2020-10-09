#!/usr/bin/python3

import string

def added_word_0(stnc1,stnc2):
    set_1 = set(stnc1.split(' '))
    set_2 = set(stnc2.split(' '))

    print(set_1)
    print(set_2)
    
    print(set_1 ^ set_2)

def added_word_1(stnc1,stnc2):
    sent_1 = stnc1.split(' ')
    sent_2 = stnc2.split(' ')

    dct_1 = {};dct_2 = {}
    for el in sent_1:
        dct_1.setdefault(el,0)
        dct_1[el] += 1

    for el in sent_2:
        dct_2.setdefault(el,0)
        dct_2[el] += 1
        
def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

    # dct = dict(zip(list(range(1,27)),[chr(el).upper() for el in range(97,123)]))

print(colnum_string(27))
    
# added_word_1('This is a dog', 'This is a fast dog')

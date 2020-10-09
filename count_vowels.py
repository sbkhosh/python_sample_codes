#!/usr/bin/python3

def cnt_vow(strg):
    dct = {}
    for el in strg:
        dct.setdefault(el,0)
        dct[el]+=1
    return(dct)

def omit_by(dct, predicate=lambda x: x=='a' or x=='e' or x=='i' or x=='o' or x=='u' or x=='y'):
    return({k:v for k,v in dct.items() if predicate(k)})


if __name__ == '__main__':
    strg = 'listening'
    res = cnt_vow(strg)
    print(omit_by(res))

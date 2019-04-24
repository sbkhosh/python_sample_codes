#!/usr/bin/python3

def makeAnagram(a,b):
    a_set = set(a);b_set = set(b)
    dct_cnt_a = {};dct_cnt_b = {}

    for el in a_set:
        dct_cnt_a[str(el)] = list(a).count(el)
    for el in b_set:
        dct_cnt_b[str(el)] = list(b).count(el)

    sym_diff = a_set ^ b_set
    inter = a_set.intersection(b_set)

    count_a_sym = count_b_sym = 0
    count_a_inter = count_b_inter = 0
    
    for el in sym_diff:
        if(el in dct_cnt_a.keys()):
            count_a_sym += dct_cnt_a[str(el)]
        elif(el in dct_cnt_b.keys()):
            count_b_sym += dct_cnt_b[str(el)]
    sum_sym = count_a_sym+count_b_sym

    for el in inter:
        count_a_inter += dct_cnt_a[str(el)]
        count_b_inter += dct_cnt_b[str(el)]
    sum_inter = abs(count_a_inter-count_b_inter)

    return(sum_sym + sum_inter)
    
        
                    
a = "tsaae"# input()
b = "aadeedaf"# input()
res = makeAnagram(a, b)
print(res)

#!/usr/bin/python3

def def_round(x):
    if (x < 0.0):
        return int(x - 0.5)
    else:
        return int(x + 0.5)

def post_decimal(x):
    num = []
    count = 0
    residue = x - int(x)
    if residue != 0:
        multiplier = 1
        while not (x*multiplier).is_integer():
            count += 1
            multiplier = 10 * multiplier
            num.append(def_round(x*multiplier)/10**count)
    return(num)
            
if __name__ == '__main__':
    n = 123.4561123
    n2round = 1
    idx = n2round - 1

    all_rounds = post_decimal(n)
    print(all_rounds[idx])

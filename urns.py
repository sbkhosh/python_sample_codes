#!/usr/bin/python3

def pwin():
    nb1_res, nr1_res, prb_res = [],[],[]
    for nb1 in range(6):
        for nr1 in range(6):
            if( (nb1 == 0 and nr1 == 0) or (nb1 == 5 and nr1 == 5)):
                p=0
            else:
                p = 0.5*(nb1/(nb1+nr1)) + 0.5*((5-nb1)/(10-(nb1+nr1)))
            nb1_res.append(nb1)
            nr1_res.append(nr1)
            prb_res.append(p)

    return(list(zip(prb_res,nb1_res,nr1_res)))
            
if __name__ == '__main__':
    res = pwin()
    print(max(res))

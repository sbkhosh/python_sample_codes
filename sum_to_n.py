#!/usr/bin/python3

def divs(num):
    res = [ x  for x in range(1,num+1) if(num%x==0) ][:-1]
    return(res)

def solution(lst,target):
    res = []
    for el in enumerate(lst):
        if(target % el == 0):
            res.append([el]*(target//el))
        
        # else:
        #     if(target-lst[i-1] % el):
        #         res.append( [lst[i-1]] + [el] * ((target-lst[i-1]) // el))
    return(res)
    
if __name__ == '__main__':
    print(solution([2,3,5],10))
    

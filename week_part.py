#!/usr/bin/python3

import pandas as pd

def fnc(num):
    if(num<=7):
        return(1)
    elif(7<num<=14):
        return(2)
    elif(14<num<=21):
        return(3)
    elif(21<num):
        return(4)

def solution(lst):
    df = pd.DataFrame(pd.to_datetime(lst),columns=['Dates'])
    df['year'] = [el.year for el in df['Dates']]
    df['month'] = [el.month for el in df['Dates']]
    df['day'] = [el.day for el in df['Dates']]
    df['select'] = df['day'].apply(fnc)
    dg = df.groupby('select').get_group(2)
    
if __name__ == '__main__':
    ts = ['2019-01-01', 
          '2019-01-02',
          '2019-01-08', 
          '2019-02-01', 
          '2019-02-02',
          '2019-02-05']

    solution(ts)
    

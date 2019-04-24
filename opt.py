#!/usr/bin/python

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

Nsample = 1000
NumCandidates = 3
lbound = 0
hbound = 1

def choices(Nsample, NumCandidates):
    data = random.sample(range(1, Nsample), NumCandidates)
    res = [ float(data[i])/Nsample for i in range(len(data)) ]
    res[0] = res[0] * 0
    return(res)

def ref_choice(lbound,hbound):
    ch_REF = round(random.uniform(lbound,hbound),2)
    return(ch_REF)

def mapped():   
    ch_ABC_str = ["choice A", "choice B", "choice C"]
    ch_ABC = choices(Nsample, NumCandidates)
    ch_ref = ref_choice(lbound,hbound)
    diffs = [ abs(ch_ABC[x]-ch_ref) for x in range(len(ch_ABC)) ]
    map_choices = dict(zip(ch_ABC_str, ch_ABC))
    map_diffs = dict(zip(ch_ABC_str, diffs))
    min2ref = min(zip(map_diffs.values(),map_diffs.keys()))
    return(ch_ref,min2ref[0],min2ref[1])
    # min2ref_key = min(zip(map_diffs.values(),map_diffs.keys()))[1]
    # return(ch_ref, map_choices, map_diffs, min2ref_key, min2ref_val)

def format_res(data):
    print "######################################################################################################"
    print "ref choice = ", data[0]
    print "######################################################################################################"
    print "choices =>", data[1]
    print "######################################################################################################"
    print "diffs => ", data[2]
    print "######################################################################################################"    
    print "winner is: ", data[3], "with value = ", data[4]
    print "######################################################################################################"    

def plot_data(df,var):
    df[str(var)].plot()
    plt.show()

def choice_select(df,choice):
    ch = df.loc[df["ChoiceName"] == str(choice)]
    return(ch)

def data_desc(df,index):
    print(df[index].mean())

  
Ntest = 100000
res = [ mapped() for x in range(Ntest) ]
df = pd.DataFrame(res, columns = ["Val_Ref", "Value", "ChoiceName"])
choices = ["choice A", "choice B", "choice C"]
ch_ABC = [ choice_select(df,choices[i]) for i in range(len(choices)) ]
data_desc(ch_ABC,0)
print("###################")
data_desc(ch_ABC,1)
print("###################")
data_desc(ch_ABC,2)



# df.plot.bar(x='ChoiceName', y='Value', rot=0)
# plt.show()

# df.set_index("ChoiceName", inplace=True)

# stats = df["Value"].describe()
# print(stats)

# for i in range(len(res)):
#     print(res[i])
    
    # ch_A = round(random.uniform(lbound,hbound),2)
    # ch_B = round(random.uniform(lbound,hbound),2)
    # ch_C = round(random.uniform(lbound,hbound),2)
    # assert(ch_B != ch_A)
    # assert(ch_C != ch_A and ch_C != ch_B)


# header = df.iloc[0]
# df = df[1:]
# print(df)
# df.rename(columns = header)

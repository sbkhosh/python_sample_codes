#!/usr/bin/python3

# group in 5-minute chunks. 
t = df.groupby(pd.Grouper(freq='5Min')).agg({"ZARJPY_open": "first", 
                                             "ZARJPY_close": "last", 
                                             "ZARJPY_low": "min", 
                                             "ZARJPY_high": "max"})

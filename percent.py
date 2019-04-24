#!/usr/bin/python

# a=1235.0
# per=40.0

# red=a*per/100.0
# res=a*(1.0-40./100.0)


# print "discount is : %8.6f" % red
# print "new price is : %8.6f" % res

def price_calc(x,y):
    dis=x*y/100.0
    price=x*(1.0-y/100.0)
    print "discount is : %8.6f" % dis
    print "discount is : %8.6f" % price

price_calc(1235,40)

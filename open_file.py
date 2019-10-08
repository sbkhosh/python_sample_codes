#!/usr/bin/python3

f = open('ex.txt','r')
lines = f.readlines()
print(lines)
# for line in f.readlines():
# 	print line
# f.close()


#import sys
#try:
#        fi = open('ex.txt', 'r')
#except IOError:
#        print 'Can\'t open file for reading.'
#        sys.exit(0)
#
#fi.readline('ex.txt')

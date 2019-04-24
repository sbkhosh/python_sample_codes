#!/usr/bin/python

# x = int(raw_input("Please enter an integer: "))
# str=['a','b','c']
# print len(str)
# print str[0],str[1]
# import urllib
# f = urllib.urlopen("http://www.cplusplus.com/reference/algorithm/count/")
# print f.read()
# for i in range(10):
#     print i,
# def fib(n):
#     res=[]
#     a,b=0,1
#     while a<n:
#         res.append(a)
#         a,b = b,a+b
#     return res
# print fib(100)
# f = open('/home/khosh/test/data.txt', 'r')
# s=[]
# for i in f:
#     s.append(i)
# N=len(s)
# print N
# for i in range(1,N):
#     print s[i]

# class Complex:
#     def __init__(self, realpart, imagpart):
#         self.r=realpart
#         self.i=imagpart
#     def f(self):
#         return "hello"
# x=Complex(3,4)
# print x.r, x.i

# import random
# a=random.sample(xrange(30), 30)
# b=random.sample(xrange(30), 30)

# N=len(a)
# for i in range(1,N):
#     print a[i],"*",b[i]
#     x = int(raw_input("res =  "))
#     if x==(a[i]*b[i]):
#         print "correct"
#     else:
#         print "not correct"

# from Tkinter import *

# root = Tk()

# w = Label(root, text="Hello, world!")
# w.pack()

# root.mainloop()

# from Tkinter import *

# class App:

#     def __init__(self, master):

#         frame = Frame(master)
#         frame.pack()

#         self.button = Button(frame, text="QUIT", fg="red", command=frame.quit)
#         self.button.pack(side=LEFT)

#         self.hi_there = Button(frame, text="Hello", command=self.say_hi)
#         self.hi_there.pack(side=LEFT)

#     def say_hi(self):
#         print "hi there, everyone!"

# root = Tk()

# app = App(root)

# root.mainloop()

# from Tkinter import *
# class Application(Frame):
#     def __init__(self, master=None):

#         Frame.__init__(self, master)
#         self.grid()
#         self.createWidgets()
#     def createWidgets(self):
#         self.quitButton = Button ( self, text='Quit',
#                                        command=self.quit )
#         self.quitButton.grid()
# app = Application()
# app.master.title("Sample application")
# app.mainloop()

f=open('sp500.txt','r+')
for line in f:
    print line

f.write('test\n')

for line in f:
    print line





# Fibonacci numbers module

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while b < n:
        print b,
        a, b = b, a+b

def fib2(n): # return Fibonacci series up to n
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a+b
    return result

if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))

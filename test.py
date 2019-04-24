#!/usr/bin/python

from Tkinter import *

# from Tkinter import *
# base = Tk()
# a = Label(base, text="Welcome !",fg='red')
# a.pack()
# button = Button(base, text="Quit", command=base.destroy)
# button.pack()
# base.mainloop()

# class vector(object) :
#     def __init__(self,x,y):
#         self.x=x;
#         self.y=y
#     def __add__(self,a):
#         return vector(self.x+a.x,self.y+a.y)
#     def __str__(self):
#         return "vector(%g,%g)" % (self.x,self.y)

# v1=vector(1.2,3.4)
# v2=vector(4.5,6.5)

# print v1+v2
# class C(object):
#     def __init__(self,n):
#         self.x=n
# a=C(10)
# print a.x

# f=open("yahoo.txt","r")
# s=f.read()
# for line in s:
#     print line
# f.close()

# f=open("test.txt","w")
# s="hello\n"
# f.write(s)
# arr=['a','b','c\n']
# f.writelines(arr)
# f.close()

# from math import *
# from random import *

# L_s=2.0
# A_s=L_s*L_s
# N=10000000
# i=0
# count=0
# for i in range(N):
#     x=uniform(-1,1)
#     y=uniform(-1,1)
#     d=sqrt(x*x+y*y)
#     if d<(L_s/2.0):
#         count+=1

# ppi=A_s*count/N
# print ppi

# from math import sin
# from math import pi

# def cube(x):
#     return x**3

# def vol(r):
#     return (4.0/3.0)*pi*cube(r)

# rad=float(raw_input('r = '))
# print 'r = %f, volume = %f' %(rad,vol(rad))


# def somme(*args):
#     resultat = 0
#     for nombre in args:
#         resultat += nombre
#     return resultat


# print("give an integer")
# x=input("x = ")

# if x < 0:
#     print "x is positive"
# elif x%2:
#     print "x is positive and odd"
# else:
#     print "x is even"

# from Tkinter import *
# base = Tk()
# texte = Label(base, text="Bienvenue !", fg='red')
# texte.pack()
# bouton = Button(base, text="Quit", command=base.destroybouton.pack()
# base.mainloop()

# import numpy as np 
# from pylab import *
# import matplotlib
# from scipy.special import jn
# x = np.linspace(-5, 15, 100)

# for i in range(10):
#     y = jn(i, x)
#     plot(x, y, label='$j_%i$' % i)

# title('Bessel')
# legend()
# show()

# def table(n,x):
#     i=1
#     while i < n:
#         print i*x,
#         i+=1

# table(11,8)

# list=['vache','souris','levure','bacterie']
# i=0
# while i < len(list):
#     print list[i]
#     i+=1


# week=['mo','tu','we','th','fr','sa','su']
# for i in week:
#     print i

# for i in range(1,11):
#     print i,

# odd=range(1,23,2)

# for i in range(len(odd)):
#     print odd[i],

# from math import pi
# r=input("radius = ")
# r=float(r)
# h=input("height = ")
# h=float(h)

# V=1.0/3.0*pi*r**2*h
# print "V = %6.4f" % V

# price_ht=input("price_ht = ") 
# while price_ht > 0:
#     float(price_ht)
#     price_all=price_ht*(1.0+19.6/100.0)
#     print "price_all = %16.14f" % price_all
#     price_ht=input("price_ht = ") 

# def factorial(n): 
#     n = abs(int(n)) 
#     if n < 1: n = 1

#     if n == 1: 
#         return 1 
#     else: 
#         return n * factorial(n - 1)

# print "i\ti!"
# for i in range(10):
#     print i,"\t",factorial(i)

# S=0.0
# i=0
# n=input("n = ")
# while i <= n:
#     S+=1.0/factorial(i)
#     i+=1
# print "i = %d S = %16.15f" % (i,S)

# import math
# id=input("id = ")
# if id > 0:
#     print math.sqrt(id)
# else:
#     print "id must be >= 0"

# import math
# a=input("a = ") 
# b=input("b = ") 
# c=input("c = ") 

# if a == 0:
#     if b != 0:
#         print("\nx = {:.2f}".format(-c/b))
#     else:
#         print("\nnosolutions.")
# else:
#     delta = b**2 - 4*a*c
#     if delta > 0.0:
#         rac_delta = math.sqrt(delta)
#         print("\nx1 = {:.2f} \t x2 = {:.2f}".format((-b-rac_delta)/(2*a),(-b+rac_delta)/(2*a)))
#     elif delta < 0.0:
#         print("\nno real roots")
#     else:
#         print("\nx = {:.2f}".format(-b/(2*a)))

# i=10
# while i > 0:
#     print i*"*"
#     i-=1

# for i in range(11):
#     print i*"*"

# i=10
# while i > 0:
#     print (10-i)*" ",i*"*"
#     i-=1

# for i in range(11):
#     print (11-i)*" ",i*"*" 

# for i in range(11):
#     print (11-i)*" ",(2*i+1)*"*" 

# def fib(n):
#     if n == 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         return fib(n-1) + fib(n-2)

# for i in range(20):
#     print fib(i),

# import random
# base=random.choice(["a","t","c","g"])
# print base

# day=["mo","tu","we","th","fr","sa","su"]
# ch=input("give a day : ")
# if ch in day[0:4]:
#     print "week day"
# elif ch==day[4]:
#     print "it's Friday"
# elif ch in day[5:]:
#     print "week-end"

# seq=["A","C","G","T","T","A","G","C","T","A","A","C","G"]
# seq_new=[]
# len=len(seq)
# i=1
# while i <= len:
#     if seq[i-1]=="A":
#         seq_new.append(seq[i-1].replace("A","T"))
#     elif seq[i-1]=="T":
#         seq_new.append(seq[i-1].replace("T","A"))
#     elif seq[i-1]=="C":
#         seq_new.append(seq[i-1].replace("C","G"))
#     elif seq[i-1]=="G":
#         seq_new.append(seq[i-1].replace("G","C"))
#     i+=1
# print seq
# print seq_new

# seq=[8,4,6,5,1]
# print min(seq)


# seq=["A","R","A","W","W","A","W","A","R","W","W","R","A","G","A","R"]
# len=len(seq) 
# i=1
# cA=0
# cR=0
# cW=0
# cG=0
# while i <=len:
#     if seq[i-1]=="A":
#         cA+=1
#     elif seq[i-1]=="R":
#         cR+=1
#     elif seq[i-1]=="W":
#         cW+=1
#     elif seq[i-1]=="G":
#         cG+=1
#     i+=1    

# print "---------------------------------"
# print "cA = %d | cR = %d | cW = %d | cG = %d" %(cA,cR,cW,cG)
# print "---------------------------------"


# n=input("give an integer : ")
# while n > 1:
#     if n%2==0:
#         n/=2
#     else:
#         n=3*n+1
#     print n,

# file=open("file.txt",'r')
# a=file.read()
# for i in file:
#     print i

# def rf(fl):
#     for i in fl:
#         print i

# T = [[1],[1,1]]
# print T[0]
# print T[1]
# for i in range(2,10):
#     T.append([1])   # initialization for T[i][0]=1
#     for j in range(1, i):
#         T[i].append(T[i-1][j-1] + T[i-1][j])
#     T[i].append(1)  # initialization for T[i][i]=1
#     print T[i]

# from random import *
# i=1
# n=10
# S=0.0
# file=open('random.txt','w')
# while i<=n:
#     S=S+gauss(1,2)
#     i+=1

# print>>file, S
# file.close()


# import sys
# if len(sys.argv) != 3:
#     sys.exit("exit")
# else:
#     arg_new=[]
#     i=1
#     len=len(sys.argv)
#     while i <= len-1:
#         arg_new.append(sys.argv[i])
#         i+=1
# print arg_new

# import os
# os.system("du -hs")

# from math import sqrt
# for x in range(10,21):
#     print "| x = %d | sqrt(x) = %6.3f | " %(x,sqrt(x))

# import math
# print math.cos(math.pi)

# import time
# i=1
# len=11
# while i <= len-1:
#     print i
#     time.sleep(1)
#     i+=1

# from random import uniform
# import math
# i=1
# count=0
# n=input("iterations: ")
# while i <=n-1:
#     x=uniform(-1,1)
#     y=uniform(-1,1)
#     r=math.sqrt(x**2+y**2)
#     if r<1.0:
#         count+=1
#     i+=1
# p=count/float(n)
# pi_val=4*p
# print pi_val

# seq=['girafe', 'tigre', 'singe', 'souris']
# le=len(seq)
# i=0
# while i<=le-1:
#     print seq[i],"\t",len(seq[i])
#     i+=1

# seq=['girafe', 'tigre', 'singe', 'souris']
# seq.reverse()
# print seq
# print seq[::1]

# strg=input("give your sentence : ")
# strg=strg.lower()
# for i in range(97,123):
#     print chr(i),
# print "\r"
# for i in range(97,123):
#     print strg.count(chr(i)),

# seq=[8,3,12.5,45,25.5,52,1]
# seq_old=[8,3,12.5,45,25.5,52,1]
# seq_new=[]
# le=len(seq)
# i=le
# while i > 0:
#     min(seq)
#     seq_new.append(min(seq))
#     seq.remove(min(seq))
#     i=i-1
# print seq_new
# seq_old.sort()
# print seq_old

# import random
# for i in range(1,21):
#     base=random.choice(["a","t","c","g"])
#     print base,
    
# elements={'course':'maths','credits':9,'hours':24}
# print elements['credits']
# print elements.keys()
# print elements.values()
# print elements.has_key('hours')

# def comp(n):
#     seq=[]
#     i=1
#     while i <= n:
#         seq_tmp=input("give a letter : ")
#         seq.append(seq_tmp)
#         i+=1
#     j=1
#     seq_new=[]
#     while j <= n:
#         if seq[j-1]=="A":
#             seq_new.append(seq[j-1].replace("A","T"))
#         elif seq[j-1]=="T":
#             seq_new.append(seq[j-1].replace("T","A"))
#         elif seq[j-1]=="C":
#             seq_new.append(seq[j-1].replace("C","G"))
#         elif seq[j-1]=="G":
#             seq_new.append(seq[j-1].replace("G","C"))
#         j+=1
    
#     print "original sequence      : ", seq
#     print "complementary sequence : ", seq_new

# comp(3)

# import numpy as np 
# from pylab import *
# import matplotlib

# debut=-2*np.pi
# fin=2*np.pi
# pas=0.1
# x=np.arange(debut,fin,pas)
# y=np.cos(x)

# TF=np.fft.fft(y)
# ABSTF=np.abs(TF)

# pas_xABSTF=1/(fin-debut)
# x_ABSTF=np.arange(0,pas_xABSTF * len(ABSTF),pas_xABSTF)

# figure()
# plot(x,y)
# figure()
# plot(x_ABSTF,TF)
# show()

# class Rectangle:
#     def __init__(self, long = 0.0, larg = 0.0, coul = "blanc"):
#         self.longueur = long
#         self.largeur = larg
#         self.couleur = coul
#     def calculSurface(self):
#         print "surface = %.2f m2" %(self.longueur * self.largeur)
#     def changeCarre(self, cote):
#         self.longueur = cote
#         self.largeur = cote


# rect1=Rectangle(1,2,"noir")
# rect1.calculSurface()

# rect1.changeCarre(10)
# rect1.calculSurface()

# S=0.0
# x=int(input("x = "))
# while x>0:
#     S+=x
#     x=int(input("x = "))

# print S

# import string
# import os
# import sys

# print '{0:,d}'.format(10000000)
# s=string.Template('$who love $what')
# a=s.safe_substitute(who='I',what='money')

# startdir=os.path.join(os.path.split(sys.argv[0])[1])
# print startdir 

# S='hello'
# S_n='*'.join(S)
# S_o=S_n.split('*')
# print S_n
# print S_o
# print ''.join(S_o)

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

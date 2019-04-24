#!/usr/bin/python

from tkinter import * # widgets, constants
def msg():
    print('hello stdout...') # callback handler

top = Frame()
# make a container
top.pack()
Label(top, text="Hello world").pack(side=TOP)

widget = Button(top, text="press", command=msg)
widget.pack(side=BOTTOM)
top.mainloop()

#!/usr/bin/python3
import signal
import sys

def alarm_handler(signal, frame):
    print 'BOOM!'
    sys.exit(0)
signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(5) # generate SIGALRM after 5 secs
n = 0
while 1:
    print n
    n = n+1

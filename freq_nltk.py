#!/usr/bin/python3

import nltk
  
def get_freq(tokens):
    freq = nltk.FreqDist(tokens)
    return(freq)

if __name__ == '__main__':
    x = "The war owes its historical significance to multiple factors"    
    freq = get_freq(x)
    freq.plot(100,cumulative=False)


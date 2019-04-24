#!/usr/bin/env python3

import urllib.request

x = urllib.request.urlopen('https://www.google.com/')
print(x.read())

#!/usr/bin/python

import urllib2
# urllib2 is use to download the html content of the web link
response = urllib2.urlopen('http://python.org/')
# You can read the entire content of a file using read() method
html = response.read()
print(html)

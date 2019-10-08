#!/usr/bin/python3

import os
from os.path import basename
from datetime import datetime

file = 'test.txt'

# f = open(file,'r')
# print(f.name)
# print(f.mode)
# f.close()

# with open(file,'r') as f:
#     pass
# print(f.closed())
# print(f.read())

# with open(file,'r') as f:
#     f_contents = f.read()
#     print(f_contents)

# with open(file,'r') as f:
#     f_contents = f.readline()
#     print(f_contents, end='')

# with open(file,'r') as f:
#     for line in f:
#         print(line, end='')

# with open(file,'r') as f:
#     size = 10
#     f_contents = f.read(size)
#     while len(f_contents) > 0:
#         print(f_contents, end='*')
#         f_contents = f.read(size)

# with open(file,'r') as f:
#     size = 10

#     f_contents = f.read(size)
#     print(f_contents, end='')

#     f.seek(0)
   
#     f_contents = f.read(size)
#     print(f_contents, end='')

#     print(f.tell())

# mod_time = os.stat('test2.txt').st_mtime
# print(datetime.fromtimestamp(mod_time))

# with open(file,'w') as f:
#     f.write('test')
#     f.seek(0)
#     f.write('test')

# base = os.path.splitext(file)[0]
# ext = ".txt"
# base_copy = base + "_copy" 
# file_copy = base_copy + ext
# print(file)
# print(file_copy)

# with open(file,'r') as rf:
#     with open( file_copy ,'w') as wf:
#         for line in rf:
#             wf.write(line)

# with open("stinkbug.png",'rb') as rf:
#     with open( "stinkbug_copy.png" ,'wb') as wf:
#         for line in rf:
#             wf.write(line)

with open("stinkbug.png",'rb') as rf:
    with open( "stinkbug_copy.png" ,'wb') as wf:
        chunk_size = 4096
        rf_chunk = rf.read(chunk_size)

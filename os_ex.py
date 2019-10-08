#!/usr/bin/python3

import os
import glob
from os.path import basename
from datetime import datetime

# view directories
tmp = dir(os)
if 'chdir' in tmp:
    print("True")
    
os.chdir('/home/')
lst = [ el for el in os.listdir() if '.' not in el ]
print(lst)

os.mkdir('test_dir')
os.makedirs('test_dir/sub')

os.removedir('test_dir')
os.removedirs('test_dir/sub')

# get csv files
ext = ".csv"
csv_files = []
for file in glob.glob("*.csv"):
    csv_files.append(file)

# rename --> lower/upper    
for el in csv_files:
    base = os.path.splitext(el)[0]
    os.rename(base + ext, base.lower() + ext)

# output
for file in glob.glob("*.csv"):
    print(file)

# last time modified    
mod_time = os.stat('os_ex.py').st_mtime
print(datetime.fromtimestamp(mod_time))

# traverse direcotries
for dirpath, dirname, filenames in os.walk('/home/'):
    print('CurrPath = ', dirpath)
    print('Directories = ', dirname)
    print('Files = ', filenames)

# write to file
file_path = os.path.join(os.environ.get('HOME'), 'test.txt')
with open(file_path, 'w') as f:
    f.write('test\n')
f.close()

with open(file_path, 'r') as f:
    tmp = f.readline()
print(tmp)

print(os.path.basename('tmp/test.txt'))
print(os.path.dirname('tmp/test.txt'))
print(os.path.split('tmp/test.txt'))
print(os.path.exists('tmp/test.txt'))
print(os.path.isdir('tmp/test.txt'))
print(os.path.isfile('tmp/test.txt'))
print(os.path.splitext('tmp/test.txt')[0])


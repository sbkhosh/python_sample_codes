#!/usr/bin/env python3

import fileinput
import os
import json
import shutil
import errno

def parse_input(str1,str2):
    filename = str1
    lines = [line.rstrip("\n") for line in open(filename)]
    paths = [lines[i].split("=")[1].strip() for i in range(len(lines))]
    basenames = [os.path.basename(os.path.normpath(paths)[i]) for i in range(len(paths))]
    return(dict(zip(paths,basenames)))

def rw_dat(str1,var,option):
    if(option == 'w'):
        with open(str1, 'w') as fptxt:
            json.dump(var,fptxt)
    elif(option == 'r'):
        lines = [line.rstrip('\n') for line in open(str1)]
        with open(str1) as json_file:
            data = json.load(json_file)
    else:
        print("No correct option selected")
            
    
        

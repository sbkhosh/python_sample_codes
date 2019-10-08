#!/usr/bin/python3

import hashlib

algo_avail = hashlib.algorithms_available
algo_guart = hashlib.algorithms_guaranteed

lst = [ el for el in algo_guart ] 

hash_object = hashlib.sha256(b'Hello World')
hex_dig = hash_object.hexdigest()
print(hex_dig)

# lst = [f for _, f in hashlib.__dict__.iteritems() ]
# print(lst)

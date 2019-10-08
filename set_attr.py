#!/usr/bin/python3

class Person():
    pass

person = Person()
person_info = {'first': 'S', 'last': 'K'}

for k,v in person_info.items():
    setattr(person,k,v)

print(person.__dict__)

#!/usr/bin/python3

class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@gmail.com'

    def fullname(self):
      return '{} {}'.format(self.first, self.last)
        

emp_1 = Employee('Corey', 'Schaf', 100000)
emp_2 = Employee('John', 'Doe', 200000)

Employee.fullname(emp_1) # is equivalent to below
emp_1 = Employee('Corey', 'Schaf', 100000)

print(emp_1.fullname())
print(Employee.fullname(emp_1))








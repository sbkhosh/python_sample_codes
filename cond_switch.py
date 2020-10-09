#!/usr/bin/python3

def dispatch_dict(operator,x,y):
    return {
            'add': lambda: x+y,
            'sub': lambda: x-y,
            'mul': lambda: x*y,
            'div': lambda: x/y,
            }.get(operator, lambda: None)()

if __name__ == '__main__':
    print(dispatch_dict('mul',2,3))
    

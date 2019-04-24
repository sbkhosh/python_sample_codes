#!/usr/bin/python

lst = []

n = int(raw_input())
for _ in range(n):
    line = raw_input().split()
    action_raw = line[0]
    if(action_raw == "insert"):
        action, inputs = line[0], line[1:]
        inputs = map(int, inputs)
        lst.insert(inputs[0],inputs[1])
    elif(action_raw == "append"):
        action, inputs = line[0], line[1:]
        inputs = map(int, inputs)
        lst.append(inputs[0])
    elif(action_raw == "remove"):
        action, inputs = line[0], line[1:]
        inputs = map(int, inputs)
        lst.remove(inputs[0])
    elif(action_raw == "print" or action_raw == "sort" or action_raw == "reverse" or action_raw == "pop"):
        action = line[0]
        if(action == "print"):
            print(lst)
        elif(action == "sort"):
            lst.sort()
        elif(action == "reverse"):
            lst.sort(reverse=True)
        elif(action == "pop"):
            lst.pop()
            


# [1, 48, 75, 30, 44, 6, 10, 44, 8, 9, 87, 75, 21, 2, 67, 12, 7, 66, 3, 5]
[5, 3, 66, 7, 12, 67, 2, 21, 75, 87, 9, 8, 44, 10, 6, 44, 30, 75, 48, 1]
# [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 21, 30, 44, 44, 48, 66, 67, 75, 75, 87]
# [1, 3, 5, 6, 7, 8, 9, 10, 12, 21, 30, 44, 44, 48, 66, 67, 75, 75, 87, 2, 5]

# [1, 48, 75, 30, 44, 6, 10, 44, 8, 9, 87, 75, 21, 2, 67, 12, 7, 66, 3, 5]
[87, 75, 75, 67, 66, 48, 44, 44, 30, 21, 12, 10, 9, 8, 7, 6, 5, 3, 2, 1]
# [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 21, 30, 44, 44, 48, 66, 67, 75, 75, 87]
# [1, 3, 5, 6, 7, 8, 9, 10, 12, 21, 30, 44, 44, 48, 66, 67, 75, 75, 87, 2, 5]


# 29
# append 1
# append 6
# append 10
# append 8
# append 9
# append 2
# append 12
# append 7
# append 3
# append 5
# insert 8 66
# insert 1 30
# insert 6 75
# insert 4 44
# insert 9 67
# insert 2 44
# insert 9 21
# insert 8 87
# insert 1 75
# insert 1 48
# print
# reverse
# print
# sort
# print
# append 2
# append 5
# remove 2
# print

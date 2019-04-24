#!/usr/bin/python

import numpy as np

# marksheet = [] 
# for _ in range(int(raw_input())):
#     name = raw_input()
#     score = float(raw_input())
#     marksheet.append([name, score])

# second_highest = sorted(list(set([marks for name, marks in marksheet])))[1]
# print('\n'.join([a for a,b in sorted(marksheet) if b == second_highest]))

#######################################################################
# n = int(raw_input())
# student_marks = {}
# for _ in range(n):
#     line = raw_input().split()
#     name, scores = line[0], line[1:]
#     scores = map(float, scores)
#     student_marks[name] = scores
# query_name = raw_input()
# print(np.mean(np.array(student_marks.values())))

#######################################################################
# if __name__ == '__main__':
#     n = int(raw_input())
#     student_marks = {}
#     for _ in range(n):
#         line = raw_input().split()
#         name, scores = line[0], line[1:]
#         scores = map(float, scores)
#         student_marks[name] = scores
#     query_name = raw_input()

# res = 0
# lst_query = student_marks[str(query_name)]
# for el in lst_query:
#     res += el

# print(format(res/len(lst_query), '.2f'))
###################################################################
from operator import itemgetter

def person_lister(f):
    def inner(people):
        return(map(f,people.sort(key=itemgetter(2))))
        # return people.sort(key=itemgetter(0)) # map(f, sorted(people, key=lambda x: x[2]))          
    return inner

# map(f, sorted(people, key=lambda x: x[2]))          

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [raw_input().split() for i in range(int(raw_input()))]
    print ''.join(name_format(people))

# print(np.mean(np.array(student_marks[query_name].values()))) 

# lst = [['aaa', 43.1], ['bbb', 33.1], ['ccc', 23.1], ['ddd', 23.1]]  
# dict_lst = { k[0]: k[1] for k in lst }

# sorted_dict_lst = []
# for w in sorted(dict_lst, key=dict_lst.get, reverse=True):
#     sorted_dict_lst.append([w,dict_lst[w]])

# print(sorted_dict_lst)    
# sec_index = -2    
# if(sorted_dict_lst[sec_index-1][0] == sorted_dict_lst[sec_index][0]):
#     print(sorted_dict_lst[sec_index-1][0],sorted_dict_lst[sec_index][0])
# # print(sorted_dict_lst[sec_index-1],sorted_dict_lst[sec_index],sorted_dict_lst[sec_index+1])

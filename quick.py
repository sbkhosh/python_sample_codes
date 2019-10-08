#!/usr/bin/python3

import numpy as np
import multiprocessing as mp

# (1) swap rows and colums in matrix (without np transpose)
mat = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

list(zip(*mat))

# (2) flatten any list/array
lst = [[1,2,3],[4,5,6,],[7,8,9]]
[ y for x in lst for y in x  ]

# (3) gcd
def gcd(m,n):
    while(m%n) != 0:
        old_m = m
        old_n = n

        m = old_n
        n = old_m%old_n
    return(n)

# (4) numpy stuff
a = np.arange(10)
np.where(a < 5, a, 10*a)
        

# (5) check two words are anagrams
str1,str2 = 'earth','heart'
print(sum(list(map(ord,str1))))

# (6) protect password at screen
from getpass import getpass

psswd = getpass('give your password: ')
print(psswd)

# (7) count letters
def count_letters(word):
    letters = list(word)
    set_letters = set(letters)
    dct_letters = {}
    for el in set_letters:
        dct_letters[str(el)] = letters.count(el)
    return(dct_letters)

# (8) scatter plot
def scatter_plot(df):
    scatter_matrix(df,alpha = 0.3,figsize = (6,6),diagonal = 'kde')
    plt.show()

# (9) quick plot
def plots():
    df = pd.read_csv('dudybar.txt', sep=" ", header=None)
    df.columns = ["time", "up", "down"]
    plt.scatter(x=df["up"], y=df["down"],alpha=0.3,s=30)
    plt.xlabel('up')
    plt.ylabel('down')
    plt.title('correlation')
    plt.show()

# (10) list & dict
lst = [['ddd', 23.1], ['bbb', 33.1], ['aaa', 43.1], ['ccc', 23.1]]  
dict_lst = { k[0]: k[1] for k in lst }
print(dict_lst)

sorted_dict_lst = []
for w in sorted(dict_lst, key=dict_lst.get, reverse=True):
    sorted_dict_lst.append([w,dict_lst[w]])
print(sorted_dict_lst)    

# (11) plot/subplot
plt.figure(figsize=(10,15))
ax1 = plt.subplot(2, 1, 1)
ax1.set_title(myf_csv+'- close',fontsize=16)
df['Close'].plot()

ax2 = plt.subplot(2, 1, 2, sharex = ax1)
ax2.set_title(myf_csv+'- vol',fontsize=16)
df['STD'].plot()

# (12) timing with argument
def timeit1():
    s = time.time()
    for i in xrange(750000):
        z=i**.5
    print "Took %f seconds" % (time.time() - s)

def timeit2(arg=math.sqrt):
    s = time.time()
    for i in xrange(750000):
        z=arg(i)
    print "Took %f seconds" % (time.time() - s)

# (13) L2 norm
x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)

# (14) create dataframe
data = {'name' : ['AA', 'IBM', 'GOOG'],
        'date' : ['2001-12-01', '2012-02-10', '2010-04-09'],
        'shares' : [100, 30, 90],
        'price' : [12.3, 10.3, 32.2]}

df = pd.DataFrame(data)
df = df.set_index(['date'])
df = df.drop(['shares','price'], axis = 1)

# (15) open file and read
with open('test.txt','r') as f:
    fcontents = f.read()

words = fcontents.split(' ')
wc = len(words)
print('number of words = {}'.format(wc))

# (16) count number of cpus
num_cores = mp.cpu_count()
print(num_cores)

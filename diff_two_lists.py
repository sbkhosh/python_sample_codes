#!/usr/bin/python3

diff = []
zipped = zip(reframed[reframed.columns[-1]].values,target_var)
for el1,el2 in zipped:
    diff.append(el1-el2)
print(np.sum(diff))

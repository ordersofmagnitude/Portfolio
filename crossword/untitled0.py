# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:20:54 2022

@author: nikki
"""

if var in assignment.keys():
    print("variable already assigned!")
    pass

count = {}
for value in self.domains[var]:
    print(value)
    count[value] = 0
    for nb in self.crossword.neighbors(var):
        print(nb)
        if value in self.domains[nb] and nb not in assignment.keys():
            count[value] += 1
            
for value in self.domains[var]:
    for nb in self.crossword.neighbors(var):
        if nb not in assignment.keys():
            i, j = self.crossword.overlaps[var, nb]
            for nb_value in self.crossword.neighbors(var):
                if value[i] != nb_value[j]:
                    count[value] += 1
                    
print(count)
                    
return sorted()

unassigned = list(self.domains.keys() - assignment.keys())

if len(unassigned) > 1:
    return unassigned[random.randint(0, len(unassigned)-1)]
elif len(unassigned) == 1:
    return unassigned[0]
else:
    return None
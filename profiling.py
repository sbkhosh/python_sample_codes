#! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
# création d'une liste de 500 000 000 d'éléments
# (à adapter suivant la vitess de vos machines)
taille = 5000000
print "Création d'une liste avec %d élements" %( taille )
toto = range( taille )
# la variable 'a' accède à un élément de la liste
# méthode 1
start = time.time()

for i in range( len(toto) ):
    a = toto[i]
print "méthode 1 (for in range) : %.1f secondes" %( time.time() - start )
# méthode 2
start = time.time()
for ele in toto:
    a = ele
print "méthode 2 (for in) : %.1f secondes" %( time.time() - start )
# méthode 3
start = time.time()
for ele in xrange( len(toto) ):
    a = toto[i]
print "méthode 3 (for in xrange) : %.1f secondes" %( time.time() - start )
# méthode 4
start = time.time()
for idx, ele in enumerate( toto ):
    a = ele
print "méthode 4 (for in enumerate): %.1f secondes" %( time.time() - start )

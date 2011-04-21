#
#
#
#
import os
from scipy import *
import scipy.sparse as sp
import scipy.linalg as la
from qutip import *

print "======================================"
A = Qobj([0.8, 0.1, 0.1, 0.2])
print "A = \n", A
print "A isket  = \n", isket(A)
print "A isoper = \n", isoper(A)
print "A len    = \n", prod(A.shape)
print "iter     = \n", arange(0, prod(A.shape))

print "======================================"
A = Qobj([[0.8, 0.1], [0.1, 0.2]])
print "A = \n", A
print "A isket  = \n", isket(A)
print "A isoper = \n", isoper(A)


#v,d = eig(full(A).data)
#print "A eig v = \n", v[0]
#print "A eig d = \n", d[:,0]
print

print "======================================"
A = Qobj([0])
print "A = \n", A
print

print "======================================"
X,Y = meshgrid(array([0,1,2]), array([0,1,2]))
Z   = zeros(size(X))
print "X = \n", X
print "X size = \n", size(X)
print "Z = \n", Z
print "Z size = \n", size(Z)
print

print "======================================"
psi = basis(4, 2)
print "basis(4,2) = \n", psi
print


print "======================================"
psi = arange(0, 30)
print "psi = \n", factorial(psi)
#print "psi = \n", hstack([1, cumprod(psi[1:30])])
#print "diff= \n", factorial(psi) - hstack([1, cumprod(psi[1:30])])
#print "fliplr(psi) = \n", fliplr([psi])[0]
print


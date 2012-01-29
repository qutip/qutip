#
# example that shows how to make basis transform for a Qobj
#
from qutip import *

H = sigmax() + sigmaz()

print "H =\n", H

ekets, evals = H.eigenstates()

H_eig_basis = H.transform(ekets)

print "\n---\nH_eig_basis =\n", H_eig_basis
print "evals =", evals

print "<e0|H|e0> =",  H.matrix_element(ekets[0], ekets[0])
print "<e0|H|e1> =",  H.matrix_element(ekets[0], ekets[1])
print "<e1|H|e0> =",  H.matrix_element(ekets[1], ekets[0])
print "<e1|H|e1> =",  H.matrix_element(ekets[1], ekets[1])

print "\n---\nsigma_x in eigenbasis =\n", sigmax().transform(ekets)

print "\n---\nsigma_y in eigenbasis =\n", sigmay().transform(ekets)

print "\n---\nsigma_z in eigenbasis =\n", sigmaz().transform(ekets)


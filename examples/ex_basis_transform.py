#
# example that shows how to make basis transform for a Qobj
#
from qutip import *

H = (rand()-0.5)*sigmax() + (rand()-0.5)*sigmaz()

print "H =\n", H

evals, ekets = H.eigenstates()

print "ekets =", type(ekets)

H_eig_basis = H.transform(ekets)

print "\n---\nH_eig_basis =\n", H_eig_basis
print "\n---\nekets =\n", ekets
print "evals =", evals
print "<e0|H|e0> =",  H.matrix_element(ekets[0], ekets[0])
print "<e0|H|e1> =",  H.matrix_element(ekets[0], ekets[1])
print "<e1|H|e0> =",  H.matrix_element(ekets[1], ekets[0])
print "<e1|H|e1> =",  H.matrix_element(ekets[1], ekets[1])
print "\n---\nsigma_x in eigenbasis =\n", sigmax().transform(ekets)
print "\n---\nsigma_y in eigenbasis =\n", sigmay().transform(ekets)
print "\n---\nsigma_z in eigenbasis =\n", sigmaz().transform(ekets)

print "="*80
print

M = Qobj(rand(2,2) + 1j * rand(2,2))
print "\n---\nM =\n", M
print "\n---\nM in eigenbasis =\n", M.transform(ekets)
print "\n---\nM in comp.basis =\n", M.transform(ekets).transform(ekets)

print "norm =", (M - M.transform(ekets).transform(ekets)).norm()

print "="*80
print

print "M =", M
M = M.transform(ekets)
print "M =", M
M = M.transform(ekets)
print "M =", M


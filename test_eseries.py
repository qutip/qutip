from qutip import *


print "==== Creating ESERIES 1 ===="

e1 = eseries(sigmax(), -2) + eseries(sigmax(), 2)
print "e1 ="
print e1

print "==== Creating ESERIES 3 ===="

e2 = eseries(sigmaz(), -2) + eseries(sigmaz(), 2)
print "e2 ="
print e2

print "==== Multiplication ===="

print "e1 * e2 = "
e3 = e1 * e2
print estidy(e3)

print "==== expectation values ===="

psi = basis(2,0)
rho = psi * trans(psi)
rho_es  = eseries(rho, -1)
psi = basis(2,1)
rho = psi * trans(psi)
rho_es += eseries(rho, +1)

expect_es = scalar_expect(sigmaz(), rho_es)
print "ESERIES expect of sigmaz ="
print expect_es

print "esval @ [0, 0.5, 1.0, 1.5] ="
print esval(expect_es, array([0, 0.5, 1.0, 1.5]))


#
# Examples of how to use the QuTiP eseries object
#
from qutip import *
from pylab import *


print """
--------------------------------------------------------------------------------
---
--- Example eseries object: sigmax() * exp(i * omega * t)
---
"""
omega = 1.0
es1 = eseries(sigmax(), 1j * omega)

print es1

print """
--------------------------------------------------------------------------------
---
--- Example eseries object: sigmax() * cos(omega * t)
---
"""
omega = 1.0
es2 = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)

print es2


print """
--------------------------------------------------------------------------------
---
--- Evaluate eseries object at time t = 0.0
---
"""

print esval(es2, 0.0)


print """
--------------------------------------------------------------------------------
---
--- Evaluate eseries object at array of times t = [0.0, 1.0 * pi, 2.0 * pi]
---
"""
tlist = [0.0, 1.0 * pi, 2.0 * pi]
print esval(es2, tlist)


print """
--------------------------------------------------------------------------------
---
--- Expectation values of eseries
---
"""

print es2
print expect(sigmax(), es2)

print """
--------------------------------------------------------------------------------
---
--- Arithmetics with eseries
---
"""
es1 = eseries(sigmax(), 1j * omega)
print "es1 =", es1

es2 = eseries(sigmax(), -1j * omega)
print "es2 =", es2

print "===> es1 + es2 ="
print es1 + es2
print "===> es1 - es2 ="
print es1 - es2
print "===> es1 * es2 ="
print es1 * es2

print "===> (es1 + es2) * (es1 - es2) ="
print (es1 + es2) * (es1 - es2)



print "="*80




print """
--------------------------------------------------------------------------------
---
--- Expectation values of eseries
---
"""
es3 = eseries([0.5*sigmaz(), 0.5*sigmaz()], [1j, -1j]) + eseries([-0.5j*sigmax(), 0.5j*sigmax()], [1j, -1j])
print "es3 =\n", es3

print "es3 at t=0.0  =\n", es3.value(0.0)
print "es3 at t=pi/2 =\n",  es3.value(pi/2)


rho = fock_dm(2, 1)
es3_expect = expect(rho, es3)
print "Expectation value of es3 for excited spin state ="
print es3_expect

print "Expectation value at t = 0 and t = pi/2"
print es3_expect.value([0.0, pi/2])



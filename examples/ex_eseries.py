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

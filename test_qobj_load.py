from qutip import *

filename = "rho.qobj"

rho = qobj_load(filename)

print "read the following density matrix in the file " + filename
print "rho =", rho



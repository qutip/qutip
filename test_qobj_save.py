from qutip import *

filename = "rho.qobj"

rho = coherent_dm(5, 1)

print "storing the following density matrix in the file " + filename
print "rho =", rho

qobj_save(rho, filename)


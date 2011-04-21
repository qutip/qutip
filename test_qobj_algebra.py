#
#
#
#
from scipy import *
import scipy.sparse as sp
import scipy.linalg as la

from qutip.operators import destroy
from qutip.qobj import qobj
from qutip.expm import expm
from qutip.basis import basis
from qutip.wigner import wigner
from qutip.istests import *

N = 2

#A_sp = sp.lil_matrix((2,2), dtype='complex128')
A_sp = sp.lil_matrix((2,2), dtype='float32')
A_sp[0,0] = 1.0
A_sp[1,0] = 2.0
A_sp[0,1] = 3.0
A_sp[1,1] = 4.0
A = qobj(A_sp.tocsr())

#B_sp = sp.lil_matrix((2,2), dtype='complex128')
B_sp = sp.lil_matrix((2,2), dtype='float32')
B_sp[0,0] = 1.0
B_sp[1,1] = 4.0
B = qobj(B_sp.tocsr())

print "======================================"
print "A = \n", A
print
print "B = \n", B
print
print "A*B = \n", A*B
print
print "B*A = \n", B*A
print

print "======================================"
print "A = \n", A
print
print "B_full = \n", B.full()
print
print "A*B_full = \n", A*B.full()
print
print "B_full*A = \n", B.full()*A
print


print "======================================"
print "A_full = \n", A.full()
print
print "B_full = \n", B.full()
print
print "A_full*B_full = \n", A.full()*B.full()
print
print "B_full*A_full = \n", fB.full()*A.full()
print


#print "a'   = ", trans(a)
#print "a'*a = ", trans(a)*a
#print "D = ", D
#print "S = ", S
#print "D*S = ", D*S
#print "psi = ", psi

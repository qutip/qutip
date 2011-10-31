from qutip import *

# create composite operators
sx1=tensor(sigmax(),qeye(2),qeye(2))
sy1=tensor(sigmay(),qeye(2),qeye(2))

sx2=tensor(qeye(2),sigmax(),qeye(2))
sy2=tensor(qeye(2),sigmay(),qeye(2))

sx3=tensor(qeye(2),qeye(2),sigmax())
sy3=tensor(qeye(2),qeye(2),sigmay())

op1=sx1*sy2*sy3
op2=sy1*sx2*sy3
op3=sy1*sy2*sx3
opghz=sx1*sx2*sx3

# need simultaneous eigenkets of op1,op2,op3 and opghz
states,evalues=simdiag([op1,op2,op3,opghz])

# eigenvalues show contradiction with classical prediction
print evalues[:,0]
[ 1.  1.  1. -1.]

# Eigenstate is entangled superposition of up-up-up and dn-dn-dn
print states[0]
Quantum object: dims = [[2, 2, 2], [1]], shape = [8, 1], type = ket
Qobj data = 
[[-0.70710678+0.j]
 [ 0.00000000+0.j]
 [ 0.00000000+0.j]
 [ 0.00000000+0.j]
 [ 0.00000000+0.j]
 [ 0.00000000+0.j]
 [ 0.00000000+0.j]
 [ 0.70710678+0.j]]

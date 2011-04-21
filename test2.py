from qutip import *


N0=5
N1=5
N2=5
alpha=sqrt(2)
#define operators
a0=tensor(destroy(N0),qeye(N1),qeye(N2))
a1=tensor(qeye(N0),destroy(N1),qeye(N2))
a2=tensor(qeye(N0),qeye(N1),destroy(N2))


A=(alpha*a0.dag()-conj(alpha)*a0)
start_time=time.time()
D=A.expm()
finish_time=time.time()
print 'time elapsed = ',finish_time-start_time


print D



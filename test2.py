from qutip import *
import scipy.linalg as la
#import pyximport
#pyximport.install()
from sp_expm import sp_expm

N0=10
N1=10
N2=10
alpha=sqrt(6)
#define operators
a0=tensor(destroy(N0),qeye(N1),qeye(N2))
a1=tensor(qeye(N0),destroy(N1),qeye(N2))
a2=tensor(qeye(N0),qeye(N1),destroy(N2))


A=(alpha*a0.dag()-conj(alpha)*a0)
start_time=time.time()
A.expm()
finish_time=time.time()
print 'time elapsed = ',finish_time-start_time

start_time=time.time()
sp_expm(A)
finish_time=time.time()
print 'time elapsed = ',finish_time-start_time




from qutip import *
from pylab import *
import time


N0=15
N1=15
N2=15
K=1.0
gamma0=0.0
gamma1=0.0
gamma2=0.5
alpha=sqrt(5)
epsilon=0.5j #sqeezing parameter
tfinal=4.0
dt=0.05
tlist=arange(0.0,tfinal+dt,dt)
taulist=K*tlist #non-dimensional times
ntraj=50

#define operators
a0=tensor(destroy(N0),qeye(N1),qeye(N2))
a1=tensor(qeye(N0),destroy(N1),qeye(N2))
a2=tensor(qeye(N0),qeye(N1),destroy(N2))

num0=a0.dag()*a0
num1=a1.dag()*a1
num2=a2.dag()*a2

#dissipative operators for zero-temp. baths
C0=sqrt(2.0*gamma0)*a0
C1=sqrt(2.0*gamma1)*a1
C2=sqrt(2.0*gamma2)*a2


#initial state: coherent mode 0 & vacuum for modes #1 & #2
vacuum=tensor(basis(N0,0),basis(N1,0),basis(N2,0))
D=(alpha*a0.dag()-conj(alpha)*a0).expm()
psi0=D*vacuum

H=1j*K*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)
Heff=H-0.5*1j*(C0.dag()*C0+C1.dag()*C1+C2.dag()*C2)


start_time=time.time()
ops=mcsolve(Heff,psi0,taulist,ntraj,[C0,C1,C2],[num0,num1,num2])
finish_time=time.time()
print 'time elapsed = ',finish_time-start_time

avg=sum(ops,axis=0)/ntraj

plot(taulist,avg[0],taulist,avg[1],taulist,avg[2])
show()


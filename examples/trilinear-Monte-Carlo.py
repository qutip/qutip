from qutip import *
from pylab import *
import time
#number of states for each mode
N0=8
N1=8
N2=8
K=1.0
#damping rates
gamma0=0.1
gamma1=0.1
gamma2=0.4
alpha=sqrt(3)#initial coherent state param for mode 0
epsilon=0.5j #sqeezing parameter
tfinal=4.0
dt=0.05
tlist=arange(0.0,tfinal+dt,dt)
taulist=K*tlist #non-dimensional times
ntraj=100#number of trajectories
#define operators
a0=tensor(destroy(N0),qeye(N1),qeye(N2))
a1=tensor(qeye(N0),destroy(N1),qeye(N2))
a2=tensor(qeye(N0),qeye(N1),destroy(N2))
#number operators for each mode
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
#trilinear Hamiltonian
H=1j*K*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)
#run Monte-Carlo
start_time=time.time()
#avg=mcsolve(H,psi0,taulist,ntraj,[C0,C1,C2],[num0,num1,num2])
output=mesolve(H,psi0,taulist,[C0,C1,C2],[num0,num1,num2])
avg=output.expect
finish_time=time.time()
print 'time elapsed = ',finish_time-start_time
#plot expectation value for photon number in each mode
plot(taulist,avg[0],taulist,avg[1],taulist,avg[2])
xlabel("Time")
ylabel("Average number of particles")
legend(('Mode 0', 'Mode 1','Mode 2') )
show()


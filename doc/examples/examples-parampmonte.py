from qutip import *
from pylab import *

N1=N2=4 #Hilbert space size 
K=0.5 #Driving strength

#damping rates
gamma1=0.1
gamma2=0.4

tfinal=16.0
taulist=linspace(0.0,tfinal,100)
ntraj=100#number of trajectories

#define operators
a1=tensor(destroy(N1),qeye(N2))
a2=tensor(qeye(N1),destroy(N2))

#number operators for each mode
num1=a1.dag()*a1
num2=a2.dag()*a2

#dissipative operators for zero-temp. baths
C1=sqrt(2.0*gamma1)*a1
C2=sqrt(2.0*gamma2)*a2

#initial state: coherent mode 0 & vacuum for modes #1 & #2
psi0=tensor(basis(N1,0),basis(N2,0))

#trilinear Hamiltonian
H=1.0j*K*(a1.dag()*a2.dag()-a1*a2)

#run Monte-Carlo
avgmc=mcsolve(H,psi0,taulist,ntraj,[C1,C2],[num1,num2])
#run Master equation
avg=odesolve(H,psi0,taulist,[C1,C2],[num1,num2])

#set legend font size
params = {'legend.fontsize': 12}
rcParams.update(params)

#plot expectation value for photon number in each mode
plot(taulist,avg[0],'k',taulist,avg[1],'k--',taulist,avgmc[0],'r',taulist,avgmc[1],'r--',lw=1.5)
xlabel("Time",fontsize=14)
ylabel("Average number of particles",fontsize=14)
legend(('Mode 1 (ME)','Mode 2 (ME)','Mode 1 (MC)','Mode 2 (MC)'),loc=2 )
show()
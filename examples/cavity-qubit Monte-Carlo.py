#########
#Monte-Carlo time evolution of an atom+cavity system.
#Adapted from a qotoolbox example by Sze M. Tan
#########
from qutip import *
from pylab import *
#inital settings
kappa=2.0 #mirror coupling
gamma=0.2 #spontaneous emission rate
g=1 #atom/cavity coupling strength
wc=0 #cavity frequency
w0=0 #atom frequency
wl=0 #driving frequency
E=0.5 #driving amplitude
N=4 #number of cavity energy levels (0->3 Fock states)
tlist=linspace(0,10,101) #times at which expectation values are needed
ntraj=500 #number of Monte-Carlo trajectories
# Hamiltonian
ida=qeye(N)
idatom=qeye(2)
a=tensor(destroy(N),idatom)
sm=tensor(ida,sigmam())
H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)
#collapse operators
C1=sqrt(2*kappa)*a
C2=sqrt(gamma)*sm
C1dC1=C1.dag()*C1
C2dC2=C2.dag()*C2
#intial state
psi0=tensor(basis(N,0),basis(2,1))
#Effective Hamiltonian 

Heff=H-0.5j*(C1dC1+C2dC2)
#run monte-carlo solver

ntraj=500

start_time=time.time()

avg=mcsolve(H,psi0,tlist,ntraj,[C1,C2],[C1dC1,C2dC2])
#plot results
plot(tlist,avg[0],tlist,avg[1],'--')
xlabel('Time')
ylabel('Photocount rates')
legend(('Cavity ouput', 'Spontaneous emission') )
show()
    



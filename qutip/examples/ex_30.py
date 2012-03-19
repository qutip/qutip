#
# Monte Carlo evoution of a coherently driven cavity with a two-level atom
# initially in the ground state and no photons in the cavity
#
#Adapted from qotoolbox example 'probqmc3' by Sze M. Tan
#
from qutip.expect import *
from qutip.mcsolve import *
from qutip.operators import *
from qutip.states import *
from qutip.tensor import *
from pylab import *

def run():
    # set system parameters
    kappa=2.0 #mirror coupling
    gamma=0.2 #spontaneous emission rate
    g=1 #atom/cavity coupling strength
    wc=0 #cavity frequency
    w0=0 #atom frequency
    wl=0 #driving frequency
    E=0.5 #driving amplitude
    N=4 #number of cavity energy levels (0->3 Fock states)
    tlist=linspace(0,10,101) #times for expectation values
    
    # construct Hamiltonian
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
    
    #run monte-carlo solver with default 500 trajectories
    start_time=time.time()
    data=mcsolve(H,psi0,tlist,[C1,C2],[C1dC1,C2dC2])
    #plot expectation values
    plot(tlist,data.expect[0],tlist,data.expect[1],lw=2)
    legend(('Transmitted Cavity Intensity','Spontaneous Emission'))
    ylabel('Counts')
    xlabel('Time')
    show()

if __name__=='__main__':
    run()



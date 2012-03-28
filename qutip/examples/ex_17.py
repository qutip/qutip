#
# Steady-state density matrix of a two-level atom in a high-Q
# cavity for various driving frequencies calculated using 
# iterative 'steady' solver.
#
# A faster version, using the parallel 'parfor' function is 
# given in the Users Guide.  
#
# Adapted from 'probss' example in the qotoolbox by Sze M. Tan.
#
from qutip.expect import *
from qutip.operators import *
from qutip.parfor import *
from qutip.states import *
from qutip.steady import *
from qutip.tensor import *
from pylab import *
import time



def probss(E,kappa,gamma,g,wc,w0,wl,N):
    #construct composite operators
    ida=qeye(N)
    idatom=qeye(2)
    a=tensor(destroy(N),idatom)
    sm=tensor(ida,sigmam())
    #Hamiltonian
    H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)
    
    #Collapse operators
    C1=sqrt(2*kappa)*a
    C2=sqrt(gamma)*sm
    C1dC1=C1.dag() * C1
    C2dC2=C2.dag() * C2
    
    #find steady state
    rhoss=steadystate(H, [C1, C2])
    
    #calculate expectation values
    count1=expect(C1dC1,rhoss)
    count2=expect(C2dC2,rhoss)
    infield=expect(a,rhoss)
    return count1,count2,infield


def run():
    # setup the calculation
    #-----------------------
    kappa=2
    gamma=0.2
    g=1
    wc=0
    w0=0
    N=5
    E=0.5
    nloop=101
    wlist=linspace(-5,5,nloop)
    #going to use lists instead of arrays here as an example.
    count1=[]
    count2=[]
    infield=[]
    
    #run loop over wlist
    for wl in wlist:
        c1,c2,infld = probss(E,kappa,gamma,g,wc,w0,wl,N)
        #append results to lists (append to array is costly)
        count1.append(c1)
        count2.append(c2)
        infield.append(infld)
    
    #plot cavity emission and qubit spontaneous emssion
    fig=figure()
    ax = fig.add_subplot(111)
    ax.plot(wlist,count1,wlist,count2,lw=2)
    xlabel('Drive Frequency Detuning')
    ylabel('Count rates')
    show()
    
    #plot phase shift of cavity light
    fig2=figure()
    ax2= fig2.add_subplot(111)
    ax2.plot(wlist,180.0*angle(infield)/pi,lw=2)
    xlabel('Drive Frequency Detuning')
    ylabel('Intracavity phase shift')
    show()

if __name__=="__main__":
    run()
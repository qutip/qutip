#
# Steady-state density matrix of a two-level atom in a high-Q
# cavity for various driving frequencies calculated using 
# iterative 'steady' solver.
#
# Adapted from 'probss' example in the qotoolbox by Sze M. Tan.
#
from qutip import *
from pylab import *


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


# setup the calculation
#-----------------------
# must be done before parfunc unless we
# want to pass all variables as one using
# zip function (see documentation for an example)
kappa=2
gamma=0.2
g=1
wc=0
w0=0
N=5
E=0.5
nloop=101
wlist=linspace(-5,5,nloop)


# define single-variable function for use in parfor
# cannot be defined inside run() since it needs to
# be passed into seperate threads.
def parfunc(wl):#function of wl only
    count1,count2,infield=probss(E,kappa,gamma,g,wc,w0,wl,N)
    return count1,count2,infield


def run():
    
    #run parallel for-loop over wlist
    count1,count2,infield = parfor(parfunc,wlist)

    #plot cavity emission and qubit spontaneous emssion
    fig=figure(1)
    ax = fig.add_subplot(111)
    ax.plot(wlist,count1,wlist,count2,lw=2)
    xlabel('Drive Frequency Detuning')
    ylabel('Count rates')
    show()
    
    #plot phase shift of cavity light
    fig2=figure(2)
    ax2= fig2.add_subplot(111)
    ax2.plot(wlist,180.0*angle(infield)/pi,lw=2)
    xlabel('Drive Frequency Detuning')
    ylabel('Intracavity phase shift')
    show()


if __name__=="__main__":
    run()
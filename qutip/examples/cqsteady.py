from ..tensor import *
from ..expect import *
from ..states import *
from ..operators import *
from ..steady import *
from ..superoperator import *
from ..parfor import *
from termpause import termpause
from pylab import *
#Adapted from the qotoolobx example 'probss' by Sze. M. Tan
#function for solving the steady-state dynamics

#-------------------------------------------------------------------------------
# setup the calculation
#-------------------------------------------------------------------------------
kappa=2            #mirror coupling
gamma=0.2      #spontaneous emission rate
g=1                    #atom-cavity coupling
wc=0                  #cavity frequency
w0=0                 #atomic frequency
N=5                    #size of Hilbert space for the cavity (zero to N-1 photons) 
E=0.5                 #amplitude of driving field
nloop=101
wlist=linspace(-5,5,nloop) #array of driving field frequency's

def probss(E,kappa,gamma,g,wc,w0,wl,N):
    ida=qeye(N)            #identity operator for cavity
    idatom=qeye(2)     #identity operator for qubit
    a=tensor(destroy(N),idatom) #destruction operator for cavity excitations for cavity+qubit system
    sm=tensor(ida,sigmam())      #destruction operator for qubit excitations for cavity+qubit system

    #Hamiltonian
    H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)

    #Collapse operators
    C1=sqrt(2*kappa)*a
    C2=sqrt(gamma)*sm
    C1dC1=C1.dag() * C1
    C2dC2=C2.dag() * C2

    #Liouvillian
    L = liouvillian(H, [C1, C2])

    #find steady state
    rhoss=steady(L)

    #calculate expectation values
    count1=expect(C1dC1,rhoss)
    count2=expect(C2dC2,rhoss)
    infield=expect(a,rhoss)
    return count1,count2,infield
    
def func(wl):#function of wl only
    count1,count2,infield=probss(E,kappa,gamma,g,wc,w0,wl,N)
    return count1,count2,infield


def cqsteady():
    print '-'*80
    print 'Solves for the steady-state dynamics of a resonantly'
    print 'driven cavity coupled to a qubit.'
    print '-'*80
    termpause()
    print 'Setup the calculation:'
    print '----------------------'
    print 'kappa=2         #mirror coupling'
    print 'gamma=0.2       #spontaneous emission rate'
    print 'g=1             #atom-cavity coupling'
    print 'wc=0            #cavity frequency'
    print 'w0=0            #atomic frequency'
    print 'N=5             #size of Hilbert space for the cavity (zero to N-1 photons) '
    print 'E=0.5           #amplitude of driving field'
    print 'nloop=101'
    print "wlist=linspace(-5,5,nloop) #array of driving field frequency's"
    
    print 'def probss(E,kappa,gamma,g,wc,w0,wl,N):'
    print '    ida=qeye(N)            #identity operator for cavity'
    print '    idatom=qeye(2)     #identity operator for qubit'
    print '    a=tensor(destroy(N),idatom) #destruction operator for cavity excitations for cavity+qubit system'
    print '    sm=tensor(ida,sigmam())      #destruction operator for qubit excitations for cavity+qubit system'

    print '    #Hamiltonian'
    print '    H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)'

    print '    #Collapse operators'
    print '    C1=sqrt(2*kappa)*a'
    print '    C2=sqrt(gamma)*sm'
    print '    C1dC1=C1.dag() * C1'
    print '    C2dC2=C2.dag() * C2'

    print '    #Liouvillian'
    print '    L = liouvillian(H, [C1, C2])'

    print '    #find steady state'
    print '    rhoss=steady(L)'

    print '    #calculate expectation values'
    print '    count1=expect(C1dC1,rhoss)'
    print '    count2=expect(C2dC2,rhoss)'
    print '    infield=expect(a,rhoss)'
    print '    return count1,count2,infield'
    
    print ''
    print 'Define single-variable function of driving frequency for use in parfor:'
    print '-----------------------------------------------------------------------'
    termpause()
    print 'def func(wl):#function of wl only'
    print '    count1,count2,infield=probss(E,kappa,gamma,g,wc,w0,wl,N)'
    print '    return count1,count2,infield'
    
    print ''
    print 'run simulation by looping over wl array in parallel using parfor'
    print '[count1,count2,infield] = parfor(func,wlist)'
    [count1,count2,infield] = parfor(func,wlist)

    print ''
    print 'Plot results...'
    termpause()
    
    print 'fig=figure()'
    print 'plot(wlist,count1,wlist,count2)'
    print "xlabel('Detuning')"
    print "ylabel('Count rates')"
    print 'show()'
    
    fig=figure()
    plot(wlist,count1,wlist,count2)
    xlabel('Detuning')
    ylabel('Count rates')
    show()
    
    close(fig)
    
    print ''
    print 'fig=figure()'
    print 'plot(wlist,180.0*angle(infield)/pi)'
    print "xlabel('Detuning')"
    print "ylabel('Intracavity phase shift')"
    print 'show()'
    fig=figure()
    plot(wlist,180.0*angle(infield)/pi)
    xlabel('Detuning')
    ylabel('Intracavity phase shift')
    show()
    print 'DEMO FINISHED...'




if __name__=='main()':
    cqsteady()
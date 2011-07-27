from scipy import *
from ..states import *
from ..operators import *
from ..tensor import *
from ..ptrace import *
from ..odesolve import *
from ..expect import *
from ..metrics import fidelity
from pylab import *
from termpause import termpause
def fidelity_example():
    print '-'*80
    print 'Here we measure the distance of a single mode (mode #1)'
    print 'of a trilinear Hamiltonian from that of a thermal density'
    print 'matrix characterized by the expectation value of the number'
    print 'of excitations in the mode at time t. Here the pump mode' 
    print '(mode #0) is assumed to be in a initial coherent state with'
    print ' the given excitation number.'
    print '-'*80
    termpause()
    
    print 'fids=zeros((3,60)) #initialize data matrix'
    print 'hilbert=[4,5,6] #list of Hilbert space sizes'
    print "num_sizes=[1,2,3] #list of <n>'s for initial state of pump mode #0 "
    
    fids=zeros((3,60)) #initialize data matrix
    hilbert=[4,5,6] #list of Hilbert space sizes
    num_sizes=[1,2,3] #list of <n>'s for initial state of pump mode #0 

    print "#loop over lists"
    print "for j in range(3):"
    print "    #number of states for each mode"
    print "    N0=hilbert[j]"
    print "    N1=hilbert[j]"
    print "    N2=hilbert[j]"
    print "    #define operators"
    print "    a0=tensor(destroy(N0),qeye(N1),qeye(N2))"
    print "    a1=tensor(qeye(N0),destroy(N1),qeye(N2))"
    print "    a2=tensor(qeye(N0),qeye(N1),destroy(N2))"
    print "    #number operators for each mode "
    print "    num0=a0.dag()*a0"
    print "    num1=a1.dag()*a1"
    print "    num2=a2.dag()*a2"
    print "    #initial state: coherent mode 0 & vacuum for modes #1 & #2"
    print "    alpha=sqrt(num_sizes[j])#initial coherent state param for mode 0"
    print "    psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))"
    print "    #trilinear Hamiltonian"
    print "    H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)"
    print "    #run odesolver"
    print "    tlist=linspace(0,3,60)"
    print "    states=odesolve(H,psi0,tlist,[],[])"
    print "    mode1=[ptrace(k,1) for k in states] #extract mode #1"
    print "    num1=[expect(num1,k) for k in states] #get <n> for mode #1"
    print "    thermal=[thermal_dm(N1,k) for k in num1] #calculate thermal matrix for <n>"
    print "    fids[j,:]=[fidelity(mode1[k],thermal[k]) for k in range(len(tlist))] #calc. fidelity"
    
    #loop over lists
    for j in range(3):
        #number of states for each mode
        N0=hilbert[j]
        N1=hilbert[j]
        N2=hilbert[j]
    
        #define operators
        a0=tensor(destroy(N0),qeye(N1),qeye(N2))
        a1=tensor(qeye(N0),destroy(N1),qeye(N2))
        a2=tensor(qeye(N0),qeye(N1),destroy(N2))
    
        #number operators for each mode
        num0=a0.dag()*a0
        num1=a1.dag()*a1
        num2=a2.dag()*a2

        #initial state: coherent mode 0 & vacuum for modes #1 & #2
        alpha=sqrt(num_sizes[j])#initial coherent state param for mode 0
        psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))

        #trilinear Hamiltonian
        H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

        #run odesolver
        tlist=linspace(0,3,60)
        states=odesolve(H,psi0,tlist,[],[])
    
        mode1=[ptrace(k,1) for k in states] #extract mode #1
        num1=[expect(num1,k) for k in states] #get <n> for mode #1
        thermal=[thermal_dm(N1,k) for k in num1] #calculate thermal matrix for <n>
        fids[j,:]=[fidelity(mode1[k],thermal[k]) for k in range(len(tlist))] #calc. fidelity

    print ''
    print 'plot the results...'
    termpause()
    print "plot(tlist,fids[0],'b',tlist,fids[1],'r',tlist,fids[2],'g',lw=1.5)"
    print "ylim([.86,1.02])"
    print "xlabel('Time',fontsize=14)"
    print "ylabel('Fidelity',fontsize=14)"
    print "title('Distance from thermal density matrix')"
    print "legend(('$\langle n\\rangle_{0}$=1','$\langle n\\rangle_{0}$=2','$\langle n\\rangle_{0}$=3'),loc=4)"
    print "show()"
    plot(tlist,fids[0],'b',tlist,fids[1],'r',tlist,fids[2],'g',lw=1.5)
    ylim([.86,1.02])
    xlabel('Time',fontsize=14)
    ylabel('Fidelity',fontsize=14)
    title('Distance from thermal density matrix')
    legend(('$\langle n\\rangle_{0}$=1','$\langle n\\rangle_{0}$=2','$\langle n\\rangle_{0}$=3'),loc=4)
    show()
    print ''
    print 'DEMO FINISHED...'


if __name__=='main()':
    fidelity_example()





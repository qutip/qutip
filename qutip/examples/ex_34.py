#
# Example showing which times and operators
# were responsible for wave function collapse
# in the monte-carlo simulation of a dissipative
# trilinear Hamiltonian.
#

from qutip import *
from pylab import *
import matplotlib.pyplot as plt

def run():
    #number of states for each mode
    N0=6
    N1=6
    N2=6
    #damping rates
    gamma0=0.1
    gamma1=0.4
    gamma2=0.1
    alpha=sqrt(2)#initial coherent state param for mode 0
    tlist=linspace(0,4,200)
    ntraj=500#number of trajectories

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
    psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))

    #trilinear Hamiltonian
    H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

    #run Monte-Carlo
    data=mcsolve(H,psi0,tlist,[C0,C1,C2],[num0,num1,num2])

    #plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs=['b','r','g'] #set three colors, one for each operator
    for k in range(ntraj):
        if len(data.col_times[k])>0:#just in case no collapse
            colors=[cs[j] for j in data.col_which[k]]#set color
            xdat=[k for x in range(len(data.col_times[k]))]
            ax.scatter(xdat,data.col_times[k],marker='o',c=colors)
    ax.set_xlim([-1,ntraj+1])
    ax.set_ylim([0,tlist[-1]])
    ax.set_xlabel('Trajectory',fontsize=14)
    ax.set_ylabel('Collpase Time',fontsize=14)
    ax.set_title('Blue = C0, Red = C1, Green= C2')
    show()

if __name__=='__main__':
    run()
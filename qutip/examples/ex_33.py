#
# Demonstrate the deviation from a thermal distribution
# for the trilinear Hamiltonian.
# 
# Adapted from Nation & Blencowe, NJP 12 095013 (2010)
#
from qutip import *
from pylab import *

def run():
    #number of states for each mode
    N0=15
    N1=15
    N2=15

    #define operators
    a0=tensor(destroy(N0),qeye(N1),qeye(N2))
    a1=tensor(qeye(N0),destroy(N1),qeye(N2))
    a2=tensor(qeye(N0),qeye(N1),destroy(N2))

    #number operators for each mode
    num0=a0.dag()*a0
    num1=a1.dag()*a1
    num2=a2.dag()*a2

    #initial state: coherent mode 0 & vacuum for modes #1 & #2
    alpha=sqrt(7)#initial coherent state param for mode 0
    psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))

    #trilinear Hamiltonian
    H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

    #run Monte-Carlo
    tlist=linspace(0,2.5,50)
    output=mcsolve(H,psi0,tlist,[],[],ntraj=1)

    #extrace mode 1 using ptrace
    mode1=[psi.ptrace(1) for psi in output.states]
    #get diagonal elements
    diags1=[k.diag() for k in mode1]
    #calculate num of particles in mode 1
    num1=[expect(num1,k) for k in output.states]
    #generate thermal state with same # of particles
    thermal=[thermal_dm(N1,k).diag() for k in num1]

    #plot results
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    colors=['m', 'g','orange','b', 'y','pink']
    x=arange(N1)
    params = {'axes.labelsize': 14,'text.fontsize': 14,'legend.fontsize': 12,'xtick.labelsize': 14,'ytick.labelsize': 14}
    rcParams.update(params)
    fig = plt.figure()
    ax = Axes3D(fig)
    for j in range(5):
        ax.bar(x, diags1[10*j], zs=tlist[10*j], zdir='y',color=colors[j],linewidth=1.0,alpha=0.6,align='center')
        ax.plot(x,thermal[10*j],zs=tlist[10*j],zdir='y',color='r',linewidth=3,alpha=1)
    ax.set_zlabel(r'Probability')
    ax.set_xlabel(r'Number State')
    ax.set_ylabel(r'Time')
    ax.set_zlim3d(0,1)
    show()

if __name__=='__main__':
    run()


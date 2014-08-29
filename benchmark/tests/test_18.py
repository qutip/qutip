from qutip import *
from numpy import *
from time import time

def test_18(runs=1):
    """
    dissipative trilinear hamiltonian
    """
    test_name='trilinear MC_F90 [3375]'
    N0=15
    N1=15
    N2=15
    gamma0=0.01
    gamma1=0.05
    gamma2=0.05
    alpha=2
    tlist=linspace(0,5,200)
    a0=tensor(destroy(N0),qeye(N1),qeye(N2));
    a1=tensor(qeye(N0),destroy(N1),qeye(N2)); 
    a2=tensor(qeye(N0),qeye(N1),destroy(N2));
    num0=a0.dag()*a0
    num1=a1.dag()*a1
    num2=a2.dag()*a2
    C0=sqrt(2*gamma0)*a0
    C1=sqrt(2*gamma1)*a1
    C2=sqrt(2*gamma2)*a2
    H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)
    psi0=tensor([coherent(N0,alpha),basis(N1,0),basis(N2,0)])
    opts=Odeoptions(gui=False)

    tot_elapsed = 0
    for n in range(runs):
        tic=time()
        mcsolve_f90(H, psi0, tlist, [C0,C1,C2],[num0,num1,num2],options=opts)
        toc=time()
        tot_elapsed += toc - tic

    return [test_name], [tot_elapsed / runs]
 

if __name__=='__main__':
    test_18()

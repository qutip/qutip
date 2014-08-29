from qutip import *
from numpy import *
from time import time

def test_8(runs=1):
    """
    Cavity+qubit monte carlo F90 equation
    """
    test_name='JC MC_F90 [20]'
    
    kappa = 2
    gamma = 0.2
    g = 1;
    wc = 0
    w0 = 0
    wl = 0
    E = 0.5;
    N = 10
    tlist = linspace(0,10,200)

    tot_elapsed = 0
    for n in range(runs):
        tic=time()
        ida = qeye(N)
        idatom = qeye(2)
        a  = tensor(destroy(N),idatom)
        sm = tensor(ida,sigmam())
        H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)
        C1=sqrt(2*kappa)*a
        C2=sqrt(gamma)*sm
        C1dC1=C1.dag()*C1
        C2dC2=C2.dag()*C2
        psi0 = tensor(basis(N,0),basis(2,1))
        mcsolve_f90(H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a],options=Odeoptions(gui=False))
        toc=time()
        tot_elapsed += toc - tic
    
    return [test_name], [tot_elapsed / runs]
 



from qutip import *
from time import time

def test_5(runs=1):
    """
    Cavity+qubit steady state
    """
    test_name='JC SS [20]'
    
    kappa=2;gamma=0.2;g=1;wc=0
    w0=0;N=10;E=0.5;wl=0

    tot_elapsed = 0
    for n in range(runs):
        tic=time()
        ida=qeye(N)
        idatom=qeye(2)
        a=tensor(destroy(N),idatom)
        sm=tensor(ida,sigmam())
        H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)
        C1=sqrt(2*kappa)*a
        C2=sqrt(gamma)*sm
        C1dC1=C1.dag() * C1
        C2dC2=C2.dag() * C2
        L = liouvillian(H, [C1, C2])
        rhoss=steady(L)
        toc=time()
        tot_elapsed += toc - tic
    
    return [test_name], [tot_elapsed / runs]
 



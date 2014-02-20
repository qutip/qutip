from qutip import *
from numpy import *
from time import time

def test_9(runs=1):
    """
    Cavity+qubit wigner function
    """
    test_name='Wigner [10]'
    
    kappa = 2; gamma = 0.2; g = 1;
    wc = 0; w0 = 0; wl = 0; E = 0.5;
    N = 10;
    tlist = linspace(0,10,200);
    ida    = qeye(N)
    idatom = qeye(2)
    a  = tensor(destroy(N),idatom)
    sm = tensor(ida,sigmam())
    H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)
    C1=sqrt(2*kappa)*a
    C2=sqrt(gamma)*sm
    C1dC1=C1.dag()*C1
    C2dC2=C2.dag()*C2
    psi0 = tensor(basis(N,0),basis(2,1))
    rho0 = psi0.dag() * psi0
    out=mesolve(H, psi0, tlist, [C1, C2], [])
    rho_cavity=ptrace(out.states[-1],0)
    xvec=linspace(-10,10,200)

    tot_elapsed = 0
    for n in range(runs):
        tic=time()
        W=wigner(rho_cavity,xvec,xvec)
        toc=time()
        tot_elapsed += toc - tic
    
    return [test_name], [tot_elapsed / runs]
 



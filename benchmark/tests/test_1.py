from qutip import *
from time import time

def test_1(runs=1):
    """
    Construct Jaynes-Cumming Hamiltonian with Nc=10, Na=2.
    """
    test_name='Qobj add [20]'
    wc = 1.0 * 2 * pi  
    wa = 1.0 * 2 * pi
    g  = 0.05 * 2 * pi
    Nc=10
    tot_elapsed = 0
    for n in range(runs):
        tic = time()
        a = tensor(destroy(Nc),qeye(2))
        sm = tensor(qeye(Nc),sigmam())
        H = wc*a.dag()*a+wa*sm.dag()*sm+g*(a.dag()+a)*(sm.dag()+sm)
        toc = time()
        tot_elapsed += toc - tic

    return [test_name], [tot_elapsed / runs]
 



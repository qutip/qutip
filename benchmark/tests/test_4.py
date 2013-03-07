from qutip import *
from time import time

def test_4(N=1.0):
    """
    Test expm with displacement and squeezing operators.
    """
    test_name='Qobj expm [20]'
    N=20
    alpha=2+2j
    sp=1.25j
    tot_elapsed = 0
    for n in range(N):
        tic=time()
        coherent(N,alpha)
        squeez(N,sp)
        toc=time()
        tot_elapsed += toc - tic

    return [test_name], [tot_elapsed / N]
 



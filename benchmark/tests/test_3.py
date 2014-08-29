from qutip import *
from time import time

def test_3(runs=1):
    """
    ptrace 6 spin operators.
    """
    test_name='Qobj ptrace [64]'
    out=tensor([sigmax(),sigmay(),sigmaz(),sigmay(),sigmaz(),sigmax()])

    tot_elapsed = 0
    for n in range(runs):
        tic=time()
        ptrace(out,[1,3,4])
        toc=time()
        tot_elapsed += toc - tic

    return [test_name], [tot_elapsed / runs]
 



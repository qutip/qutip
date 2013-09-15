from qutip import *
from time import time

def test_2(runs=1):
    """
    Tensor 6 spin operators.
    """
    test_name='Qobj tensor [64]'

    tot_elapsed = 0
    for n in range(runs):
        tic = time()
        tensor(sigmax(),sigmay(),sigmaz(),sigmay(),sigmaz(),sigmax())
        toc = time()
        tot_elapsed += toc - tic

    return [test_name], [tot_elapsed / runs]
 



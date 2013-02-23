from qutip import *
from time import time

def test_2():
    """
    Tensor 6 spin operators.
    """
    test_name='Qobj tensor [64]'
    tic=time()
    tensor(sigmax(),sigmay(),sigmaz(),sigmay(),sigmaz(),sigmax())
    toc=time()
    return [test_name], [toc-tic]
 



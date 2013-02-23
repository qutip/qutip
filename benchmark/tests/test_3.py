from qutip import *
from time import time

def test_3():
    """
    ptrace 6 spin operators.
    """
    test_name='Qobj ptrace [64]'
    out=tensor([sigmax(),sigmay(),sigmaz(),sigmay(),sigmaz(),sigmax()])
    tic=time()
    ptrace(out,[1,3,4])
    toc=time()
    return [test_name], [toc-tic]
 



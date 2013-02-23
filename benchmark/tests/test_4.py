from qutip import *
from time import time

def test_4():
    """
    Test expm with displacement and squeezing operators.
    """
    test_name='Qobj expm [20]'
    N=20
    alpha=2+2j
    sp=1.25j
    tic=time()
    coherent(N,alpha)
    squeez(N,sp)
    toc=time()
    return [test_name], [toc-tic]
 



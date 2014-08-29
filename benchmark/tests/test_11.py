from qutip import *
from numpy import *
from time import time

def test_11(runs=1):
    """
    mcsolve evolution of 4-spin chain
    """
    test_name='MC 4-spin [16]'
    N = 4# number of spins
    # uniform parameters
    h  = 1.0 * 2 * pi * ones(N) 
    Jz = 0.1 * 2 * pi * ones(N)
    Jx = 0.1 * 2 * pi * ones(N)
    Jy = 0.1 * 2 * pi * ones(N)
    # dephasing rate
    gamma = 0.01 * ones(N)
    # intial state, first spin in state |1>, the rest in state |0>
    psi_list = []
    psi_list.append(basis(2,1))
    for n in range(N-1):
        psi_list.append(basis(2,0))
    psi0 = tensor(psi_list)
    tlist = linspace(0, 10, 200)
    # Hamiltonian
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sy
        sy_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))
    # construct the hamiltonian
    H = 0    
    # energy splitting terms
    for n in range(N):
        H += - 0.5 * h[n] * sz_list[n]
    # interaction terms
    for n in range(N-1):
        H += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
        H += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
        H += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]
    # collapse operators
    c_op_list = []
    # spin dephasing
    for n in range(N):
        c_op_list.append(sqrt(gamma[n]) * sz_list[n])
    # evolve and calculate expectation values
    opts=Odeoptions(gui=False)

    tot_elapsed = 0
    for n in range(runs):
        tic=time()
        mcsolve(H, psi0, tlist, c_op_list, sz_list,options=opts)
        toc=time()
        tot_elapsed += toc - tic

    return [test_name], [tot_elapsed / runs]
 

if __name__=='__main__':
    test_11()

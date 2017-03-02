import numpy as np
from qutip import *
import qutip.settings as qset
from qutip.cy.spmatfuncs import spmv_csr
from qutip.cy.openmp.parfuncs import spmv_csr_openmp
from timeit import default_timer as timer


def _min_timer(function, *args, **kwargs):
    min_time = 1e6
    for kk in range(1000):
        t0 = timer()
        function(*args, **kwargs)
        t1 = timer()
        min_time = min(min_time, t1-t0)
    return min_time 

def _jc_liouvillian(N):
    wc = 1.0  * 2 * np.pi  # cavity frequency
    wa = 1.0  * 2 * np.pi  # atom frequency
    g  = 0.05 * 2 * np.pi  # coupling strength
    kappa = 0.005          # cavity dissipation rate
    gamma = 0.05           # atom dissipation rate
    n_th_a = 1           # temperature in frequency units
    use_rwa = 0
    # operators
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    # Hamiltonian
    if use_rwa:
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
    else:
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
    c_op_list = []

    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm)

    return liouvillian(H, c_op_list)
    

N_max = 60
dims = np.arange(2, N_max ,dtype=int)
jc1 = 1e6
jc2 = 1
jcmax = 1
jcnnz = 0
jc2_old = 0
jcmax_old = 0
jcnnz_old = 0

for N in dims:
    L = _jc_liouvillian(N).data
    vec = rand_ket(L.shape[0],0.25).full().ravel()
    jcnnz = L.nnz
    
    ser = _min_timer(spmv_csr, L.data, L.indices, L.indptr, vec)
    par = _min_timer(spmv_csr_openmp, L.data, L.indices, L.indptr, vec,2)
    print(ser/par)      
    
        
        
        
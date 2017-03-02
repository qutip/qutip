import numpy as np
from qutip import *
import qutip.settings as qset
from qutip.cy.spmatfuncs import spmv_csr
from qutip.cy.openmp.parfuncs import spmv_csr_openmp
from timeit import default_timer as timer


def _min_timer(function, *args, **kwargs):
    min_time = 1e6
    for kk in range(10000):
        t0 = timer()
        function(*args, **kwargs)
        t1 = timer()
        min_time = min(min_time, t1-t0)
    return min_time 
    

def system_bench(func, dims):
    for N in dims:
        L = func(N).data
        vec = rand_ket(L.shape[0],0.25).full().ravel()
        nnz = L.nnz
        ser = _min_timer(spmv_csr, L.data, L.indices, L.indptr, vec)
        par = _min_timer(spmv_csr_openmp, L.data, L.indices, L.indptr, vec, 2)
        ratio = ser/par
        if ratio > 1:
            break
        nnz_old = nnz 
        ratio_old = ratio  
    
    rate = (ratio-ratio_old)/(nnz-nnz_old)
    return int(rate*(nnz-nnz_old)/2.+nnz_old)


def calculate_openmp_thresh():
  jc_dims = np.arange(2,60,dtype=int)
  jc_result = system_bench(_jc_liouvillian, jc_dims)

  opto_dims = np.arange(2,60,dtype=int)
  opto_result = system_bench(_opto_liouvillian, opto_dims)

  spin_dims = np.arange(2,15,dtype=int)
  spin_result = system_bench(_spin_hamiltonian, spin_dims)
  
  return int((jc_result+opto_result+spin_result)/3.0)
  

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
    
    
def _opto_liouvillian(N):
    Nc = 5                      # Number of cavity states
    Nm = N                     # Number of mech states
    kappa = 0.3                 # Cavity damping rate
    E = 0.1                     # Driving Amplitude         
    g0 = 2.4*kappa              # Coupling strength
    Qm = 1e4                    # Mech quality factor
    gamma = 1/Qm                # Mech damping rate
    n_th = 1                    # Mech bath temperature
    delta = -0.43               # Detuning
    a = tensor(destroy(Nc), qeye(Nm))
    b = tensor(qeye(Nc), destroy(Nm))
    num_b = b.dag()*b
    num_a = a.dag()*a
    H = -delta*(num_a)+num_b+g0*(b.dag()+b)*num_a+E*(a.dag()+a)
    cc = np.sqrt(kappa)*a
    cm = np.sqrt(gamma*(1.0 + n_th))*b
    cp = np.sqrt(gamma*n_th)*b.dag()
    c_ops = [cc,cm,cp]

    return liouvillian(H, c_ops)
    
def _spin_hamiltonian(N):
    # array of spin energy splittings and coupling strengths. here we use
    # uniform parameters, but in general we don't have too
    h  = 1.0 * 2 * np.pi * np.ones(N) 
    Jz = 0.1 * 2 * np.pi * np.ones(N)
    Jx = 0.1 * 2 * np.pi * np.ones(N)
    Jy = 0.1 * 2 * np.pi * np.ones(N)
    # dephasing rate
    gamma = 0.01 * np.ones(N)

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
    return H

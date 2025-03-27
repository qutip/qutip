import numpy as np
from numpy.testing import assert_equal
import unittest
from qutip import *
from qutip.settings import settings as qset
# if qset.has_openmp:
#    from qutip.core.cy.openmp.benchmark import _spmvpy, _spmvpy_openmp


# @unittest.skipIf(qset.has_openmp == False, 'OPENMP not available.')
@unittest.skipIf(True, 'OPENMP disabled.')
def test_openmp_spmv():
    "OPENMP : spmvpy_openmp == spmvpy"
    for k in range(100):
        L = rand_herm(10,0.25).data
        vec = rand_ket(L.shape[0],0.25).full().ravel()
        out = np.zeros_like(vec)
        out_openmp = np.zeros_like(vec)
        _spmvpy(L.data, L.indices, L.indptr, vec, 1, out)
        _spmvpy_openmp(L.data, L.indices, L.indptr, vec, 1, out_openmp, 2)
        assert (np.allclose(out, out_openmp, 1e-15))

# @unittest.skipIf(qset.has_openmp == False, 'OPENMP not available.')
@unittest.skipIf(True, 'OPENMP disabled.')
def test_openmp_mesolve():
    "OPENMP : mesolve"
    N = 100
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

    n = N - 2
    psi0 = tensor(basis(N, n), basis(2, 1))
    tlist = np.linspace(0, 1, 100)
    opts = SolverOptions(use_openmp=False)
    out = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    opts = SolverOptions(use_openmp=True)
    out_omp = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    assert (np.allclose(out.expect[0],out_omp.expect[0]))
    assert (np.allclose(out.expect[1],out_omp.expect[1]))


# @unittest.skipIf(qset.has_openmp == False, 'OPENMP not available.')
@unittest.skipIf(True, 'OPENMP disabled.')
def test_openmp_mesolve_td():
    "OPENMP : mesolve (td)"
    N = 100
    wc = 1.0  * 2 * np.pi  # cavity frequency
    wa = 1.0  * 2 * np.pi  # atom frequency
    g  = 0.5 * 2 * np.pi  # coupling strength
    kappa = 0.005          # cavity dissipation rate
    gamma = 0.05           # atom dissipation rate
    n_th_a = 1           # temperature in frequency units
    use_rwa = 0
    # operators
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    # Hamiltonian
    H0 = wc * a.dag() * a + wa * sm.dag() * sm
    H1 = g * (a.dag() + a) * (sm + sm.dag())

    H = [H0, [H1,'sin(t)']]

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

    n = N - 10
    psi0 = tensor(basis(N, n), basis(2, 1))
    tlist = np.linspace(0, 1, 100)
    opts = SolverOptions(use_openmp=True)
    out_omp = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    opts = SolverOptions(use_openmp=False)
    out = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    assert (np.allclose(out.expect[0],out_omp.expect[0]))
    assert (np.allclose(out.expect[1],out_omp.expect[1]))

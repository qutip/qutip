# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import numpy as np
from numpy.testing import assert_equal, assert_, run_module_suite
import unittest
from qutip import *
import qutip.settings as qset
if qset.has_openmp:
    from qutip.cy.openmp.benchmark import _spmvpy, _spmvpy_openmp
    

@unittest.skipIf(qset.has_openmp == False, 'OPENMP not available.')
def test_openmp_spmv():
    "OPENMP : spmvpy_openmp == spmvpy"
    for k in range(100):
        L = rand_herm(10,0.25).data
        vec = rand_ket(L.shape[0],0.25).full().ravel()
        out = np.zeros_like(vec)
        out_openmp = np.zeros_like(vec)
        _spmvpy(L.data, L.indices, L.indptr, vec, 1, out)
        _spmvpy_openmp(L.data, L.indices, L.indptr, vec, 1, out_openmp, 2)
        assert_(np.allclose(out, out_openmp, 1e-15))

@unittest.skipIf(qset.has_openmp == False, 'OPENMP not available.')
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
    opts = Options(use_openmp=False)
    out = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    opts = Options(use_openmp=True)
    out_omp = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    assert_(np.allclose(out.expect[0],out_omp.expect[0]))
    assert_(np.allclose(out.expect[1],out_omp.expect[1]))


@unittest.skipIf(qset.has_openmp == False, 'OPENMP not available.')
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
    opts = Options(use_openmp=True)
    out_omp = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    opts = Options(use_openmp=False)
    out = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm], options=opts)
    assert_(np.allclose(out.expect[0],out_omp.expect[0]))
    assert_(np.allclose(out.expect[1],out_omp.expect[1]))

if __name__ == "__main__":
    run_module_suite()

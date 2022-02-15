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
from numpy.testing import assert_, run_module_suite

np.random.seed(0)

# disable the progress bar
import os

from qutip import sigmax, sigmay, sigmaz, qeye
from qutip import basis, expect
from qutip import num, destroy, create
from qutip.interpolate import Cubic_Spline
from qutip import krylovsolve
from qutip.solver import Options
from scipy.linalg import expm
from qutip.sparse import eigh
import pytest

from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj, jmat
from scipy.stats import ortho_group
import numpy as np


def h_sho(dim):
    auts = np.linspace(0, dim, dim, dtype='complex')
    H = np.diag(auts + 1 / 2)
    O = ortho_group.rvs(dim)
    A = np.matmul(O.conj().T, np.matmul(H, O))
    A = A / np.linalg.norm(A, ord=2)
    A = Qobj(A)
    return A


def h_random(dim, distribution="normal"):
    if distribution == "uniform":
        H = np.random.random([dim, dim]) + 1j * np.random.random([dim, dim])
    if distribution == "normal":
        H = np.random.normal(
            size=[dim, dim]) + 1j * np.random.normal(size=[dim, dim])
    H = (H.conj().T + H) / 2
    H = H / np.linalg.norm(H, ord=2)
    H = Qobj(H)
    return (H)


def h_ising_transverse(N: int, hx: float, hz: float, Jx: float, Jy: float,
                       Jz: float):

    hx = hx * np.ones(N)
    hz = hz * np.ones(N)
    Jx, Jy, Jz = Jx * np.ones(N), Jy * np.ones(N), Jz * np.ones(N)

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
        H += hz[n] * sz_list[n]

    for n in range(N):
        H += hx[n] * sx_list[n]

    # interaction terms
    for n in range(N - 1):
        H += -Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -Jy[n] * sy_list[n] * sy_list[n + 1]
        H += -Jz[n] * sz_list[n] * sz_list[n + 1]
    return H


def exactsolver(H, psi0, tlist: list or np.ndarray):

    _H = H.full()
    _psi0 = psi0.full()
    eigenvalues, eigenvectors = eigh(_H)

    psi_base_diag = np.matmul(eigenvectors.conj().T, _psi0)
    U = np.exp(np.outer(-1j * eigenvalues, tlist))
    psi_list = np.matmul(eigenvectors,
                         np.multiply(U, psi_base_diag.reshape(
                             [-1, 1])))  # shape=(dim, len(tlist))
    psi_list = [Qobj(psi) for psi in psi_list.T]  # thus we transpose
    return psi_list


os.environ['QUTIP_GRAPHICS'] = "NO"


class TestKrylovSolve:
    """
    A test class for the QuTiP Krylov Approximation method Solver.
    """

    def check_evolution_states(self,
                               H,
                               psi0,
                               tlist,
                               td_args={},
                               tol=5e-3,
                               krylov_dim=20):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """

        options = Options(atol=tol)

        output = krylovsolve(H,
                             psi0,
                             tlist,
                             krylov_dim,
                             e_ops=None,
                             options=options)
        exact_output = exactsolver(H, psi0, tlist)

        fidelity = [
            1 - np.abs(np.vdot(psi_k.full(), psi_exact.full()))**2
            for (psi_k, psi_exact) in zip(output.states, exact_output)
        ]

        for fid in fidelity:
            assert fid < tol, "fidelity between exact solution and krylov \
                               solution does not match"

    def check_evolution_e_ops(self,
                              H,
                              psi0,
                              tlist,
                              td_args={},
                              tol=5e-8,
                              krylov_dim=20):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """

        options = Options(atol=tol)

        #dim_j = int(H.shape[0] / 2)
        #print(dim_j)
        e_ops = [
            jmat((H.shape[0] - 1) / 2., 'x'),
            jmat((H.shape[0] - 1) / 2., 'y'),
            jmat((H.shape[0] - 1) / 2., 'z')
        ]
        output = krylovsolve(H,
                             psi0,
                             tlist,
                             krylov_dim,
                             e_ops=e_ops,
                             options=options)
        exact_output = exactsolver(H, psi0, tlist)

        jx, jy, jz = output.expect[0], output.expect[1], output.expect[2]

        jx_analytic = np.array([expect(e_ops[0], psi) for psi in exact_output])
        jy_analytic = np.array([expect(e_ops[1], psi) for psi in exact_output])
        jz_analytic = np.array([expect(e_ops[2], psi) for psi in exact_output])

        delta_jx = np.array([
            abs(_jx - _jx_analytic)
            for (_jx, _jx_analytic) in zip(jx, jx_analytic)
        ])
        delta_jy = np.array([
            abs(_jy - _jy_analytic)
            for (_jy, _jy_analytic) in zip(jy, jy_analytic)
        ])
        delta_jz = np.array([
            abs(_jz - _jz_analytic)
            for (_jz, _jz_analytic) in zip(jz, jz_analytic)
        ])

        assert_(max(delta_jx) < tol, msg="expect X not matching analytic")
        #assert_(max(abs(jy - jy_analytic)) < tol,
        #        msg="expect Y not matching analytic")
        #assert_(max(abs(jz - jz_analytic)) < tol,
        #        msg="expect Z not matching analytic")

    def test_01_states_with_constant_H_random(self):
        "krylovsolve: states with const H random"
        dim = 512
        psi0 = np.random.random(dim) + 1j * np.random.random(dim)
        psi0 = psi0 / np.linalg.norm(psi0)
        psi0 = Qobj(psi0)
        H = h_random(dim, distribution="normal")
        tlist = np.linspace(0, 20, 200)

        self.check_evolution_states(H, psi0, tlist)
        self.check_evolution_e_ops(H, psi0, tlist)

    def test_02_states_with_constant_H_ising_transverse(self):
        "krylovsolve: states with const H Ising Transverse Field"
        N = 8
        dim = 2**N
        psi0 = np.random.random(dim) + 1j * np.random.random(dim)
        psi0 = psi0 / np.linalg.norm(psi0)
        psi0 = Qobj(psi0)
        H = h_ising_transverse(N, hx=0.1, hz=0.5, Jx=1, Jy=0, Jz=1)
        tlist = np.linspace(0, 20, 200)

        self.check_evolution_states(H, psi0, tlist)
        self.check_evolution_e_ops(H, psi0, tlist)

    def test_03_states_with_constant_H_sho(self):
        "krylovsolve: states with const H SHO"
        dim = 30
        psi0 = np.random.random(dim) + 1j * np.random.random(dim)
        psi0 = psi0 / np.linalg.norm(psi0)
        psi0 = Qobj(psi0)
        H = h_sho(dim)
        tlist = np.linspace(0, 20, 200)

        self.check_evolution_states(H, psi0, tlist)
        self.check_evolution_e_ops(H, psi0, tlist)



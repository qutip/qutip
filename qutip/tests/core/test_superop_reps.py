# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:23:46 2013

@author: dcriger
"""

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
from __future__ import division

import numpy as np
import pytest

from qutip import (
    Qobj, basis, identity, sigmax, sigmay, qeye, create, rand_super,
    rand_super_bcsz, rand_dm_ginibre, tensor, super_tensor, kraus_to_choi,
    to_super, to_choi, to_kraus, to_chi, to_stinespring, operator_to_vector,
    vector_to_operator, sprepost, destroy
)
from qutip.qip.operations.gates import swap

tol = 1e-8


def assert_kraus_equivalence(a, b, tol=tol):
    assert a.shape == b.shape
    assert a.dims == b.dims
    assert a.type == b.type
    # Kraus operators may vary by a global phase.  Let's find the first
    # non-zero element, set the phase such that that element has the same phase
    # in both, and then compare.  We have to take care to find the first
    # element.
    a, b = a.full(), b.full()
    a_nz = np.nonzero(np.abs(a) > tol)
    if len(a_nz[0]) == 0:
        np.testing.assert_allclose(b, 0, atol=tol)
    a_el, b_el = a[a_nz[0][0], a_nz[1][0]], b[a_nz[0][0], a_nz[1][0]]
    assert b_el != 0
    b *= (a_el / np.abs(a_el)) / (b_el / np.abs(b_el))
    np.testing.assert_allclose(a, b)


class TestSuperopReps(object):
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """

    def test_SuperChoiSuper(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """
        superoperator = rand_super(5)
        choi_matrix = to_choi(superoperator)
        test_super = to_super(choi_matrix)
        assert choi_matrix.type == "super"
        assert choi_matrix.superrep == "choi"
        assert test_super.type == "super"
        assert test_super.superrep == "super"
        assert abs((test_super - superoperator).norm()) < tol

    def test_SuperChoiChiSuper(self):
        """
        Superoperator: Converting two-qubit superoperator through
        Choi and chi representations goes back to right superoperator.
        """
        superoperator = super_tensor(rand_super(2), rand_super(2))

        choi_matrix = to_choi(superoperator)
        chi_matrix = to_chi(choi_matrix)
        test_super = to_super(chi_matrix)
        assert choi_matrix.type == "super"
        assert choi_matrix.superrep == "choi"
        assert chi_matrix.type == "super"
        assert chi_matrix.superrep == "chi"
        assert test_super.type == "super"
        assert test_super.superrep == "super"
        assert (test_super - superoperator).norm() < tol

    def test_ChoiKrausChoi(self):
        """
        Superoperator: Convert superoperator to Choi matrix and back.
        """
        superoperator = rand_super(5)
        choi_matrix = to_choi(superoperator)
        kraus_ops = to_kraus(choi_matrix)
        test_choi = kraus_to_choi(kraus_ops)
        assert choi_matrix.type == "super"
        assert choi_matrix.superrep == "choi"
        assert test_choi.type == "super"
        assert test_choi.superrep == "choi"
        assert (test_choi - choi_matrix).norm() < tol

    def test_NonSquareKrausSuperChoi(self):
        """
        Superoperator: Convert non-square Kraus operator to Super + Choi matrix and back.
        """
        zero = np.array([[1], [0]], dtype=complex)
        one = np.array([[0], [1]], dtype=complex)
        zero_log = np.kron(np.kron(zero, zero), zero)
        one_log = np.kron(np.kron(one, one), one)
        # non-square Kraus operator (isometry)
        kraus = Qobj(zero_log @ zero.T + one_log @ one.T)
        super = sprepost(kraus, kraus.dag())
        choi = to_choi(super)
        op1 = to_kraus(super)
        op2 = to_kraus(choi)
        op3 = to_super(choi)
        assert choi.type == "super"
        assert choi.superrep == "choi"
        assert super.type == "super"
        assert super.superrep == "super"
        assert_kraus_equivalence(op1[0], kraus, tol=1e-8)
        assert_kraus_equivalence(op2[0], kraus, tol=1e-8)
        assert abs((op3 - super).norm()) < 1e-8

    def test_NeglectSmallKraus(self):
        """
        Superoperator: Convert Kraus to Choi matrix and back. Neglect tiny Kraus operators.
        """
        zero = np.array([[1], [0]], dtype=complex)
        one = np.array([[0], [1]], dtype=complex)
        zero_log = np.kron(np.kron(zero, zero), zero)
        one_log = np.kron(np.kron(one, one), one)
        # non-square Kraus operator (isometry)
        kraus = Qobj(zero_log @ zero.T + one_log @ one.T)
        super = sprepost(kraus, kraus.dag())
        # 1 non-zero Kraus operator the rest are zero
        sixteen_kraus_ops = to_kraus(super, tol=0.0)
        # default is tol=1e-9
        one_kraus_op = to_kraus(super)
        assert len(sixteen_kraus_ops) == 16
        assert len(one_kraus_op) == 1
        assert_kraus_equivalence(one_kraus_op[0], kraus, tol=tol)

    def test_SuperPreservesSelf(self):
        """
        Superoperator: to_super(q) returns q if q is already a
        supermatrix.
        """
        superop = rand_super(5)
        assert superop is to_super(superop)

    def test_ChoiPreservesSelf(self):
        """
        Superoperator: to_choi(q) returns q if q is already Choi.
        """
        superop = rand_super(5)
        choi = to_choi(superop)
        assert choi is to_choi(choi)

    def test_random_iscptp(self):
        """
        Superoperator: Randomly generated superoperators are
        correctly reported as CPTP and HP.
        """
        superop = rand_super(5)
        assert superop.iscptp
        assert superop.ishp

    @pytest.mark.parametrize(['qobj', 'hp', 'cp', 'tp'], [
        pytest.param(sprepost(destroy(2), create(2)), True, True, False),
        pytest.param(sprepost(destroy(2), destroy(2)), False, False, False),
        pytest.param(qeye(2), True, True, True),
        pytest.param(sigmax(), True, True, True),
        pytest.param(tensor(sigmax(), qeye(2)), True, True, True),
        pytest.param(0.5 * (to_super(tensor(sigmax(), qeye(2)))
                            + to_super(tensor(qeye(2), sigmay()))),
                     True, True, True,
                     id="linear combination of bipartite unitaries"),
        pytest.param(Qobj(swap(), type='super', superrep='choi'),
                     True, False, True,
                     id="partial transpose map"),
        pytest.param(Qobj(qeye(4)*0.9, type='super'), True, True, False,
                     id="subnormalized map"),
        pytest.param(basis(2, 0), False, False, False, id="ket"),
    ])
    def test_known_iscptp(self, qobj, hp, cp, tp):
        """
        Superoperator: ishp, iscp, istp and iscptp known cases.
        """
        assert qobj.ishp == hp
        assert qobj.iscp == cp
        assert qobj.istp == tp
        assert qobj.iscptp == (cp and tp)

    def test_choi_tr(self):
        """
        Superoperator: Trace returned by to_choi matches docstring.
        """
        for dims in range(2, 5):
            assert abs(to_choi(identity(dims)).tr() - dims) < tol

    def test_stinespring_cp(self, thresh=1e-10):
        """
        Stinespring: A and B match for CP maps.
        """
        def case(map):
            A, B = to_stinespring(map)
            assert abs((A - B).norm()) < thresh

        for _ in range(4):
            case(rand_super_bcsz(7))

    def test_stinespring_agrees(self, thresh=1e-10):
        """
        Stinespring: Partial Tr over pair agrees w/ supermatrix.
        """
        def case(map, state):
            S = to_super(map)
            A, B = to_stinespring(map)
            q1 = vector_to_operator(
                S * operator_to_vector(state)
            )
            # FIXME: problem if Kraus index is implicitly
            #        ptraced!
            q2 = (A * state * B.dag()).ptrace((0,))
            assert abs((q1 - q2).norm('tr')) <= thresh

        for _ in range(4):
            case(rand_super_bcsz(2), rand_dm_ginibre(2))

    def test_stinespring_dims(self):
        """
        Stinespring: Check that dims of channels are preserved.
        """
        # FIXME: not the most general test, since this assumes a map
        #        from square matrices to square matrices on the same space.
        chan = super_tensor(to_super(sigmax()), to_super(qeye(3)))
        A, B = to_stinespring(chan)
        assert A.dims == [[2, 3, 1], [2, 3]]
        assert B.dims == [[2, 3, 1], [2, 3]]

    def test_chi_choi_roundtrip(self):
        def case(qobj):
            qobj = to_chi(qobj)
            rt_qobj = to_chi(to_choi(qobj))
            np.testing.assert_allclose(rt_qobj.full(), qobj.full())
            assert rt_qobj.type == qobj.type
            assert rt_qobj.dims == qobj.dims

        for N in (2, 4, 8):
            case(rand_super_bcsz(N))

    def test_chi_known(self):
        """
        Superoperator: Chi-matrix for known cases is correct.
        """
        def case(S, chi_expected):
            chi_actual = to_chi(S)
            chiq = Qobj(chi_expected, dims=[[[2], [2]], [[2], [2]]],
                        superrep='chi')
            assert abs((chi_actual - chiq).norm('tr')) < 1e-10

        case(sigmax(), [
            [0, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        case(to_super(sigmax()), [
            [0, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        case(qeye(2), [
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        case((-1j * sigmax() * np.pi / 4).expm(), [
            [2, 2j, 0, 0],
            [-2j, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:23:46 2013

@author: dcriger
"""

from __future__ import division

from numpy import abs, pi, asarray, kron
from numpy.linalg import norm

import pytest
from qutip.qobj import Qobj
from qutip.states import basis
from qutip.operators import identity, sigmax, sigmay, qeye, create
from qutip.qip.operations.gates import swap
from qutip.random_objects import rand_super, rand_super_bcsz, rand_dm_ginibre
from qutip.tensor import tensor, super_tensor
from qutip.superop_reps import (
    kraus_to_choi, to_super, to_choi, to_kraus, to_chi, to_stinespring,
)
from qutip.superoperator import (
    operator_to_vector, vector_to_operator, sprepost,
)

tol = 1e-9


@pytest.fixture(scope="function", params=[2, 3, 7])
def dimension(request):
    # There are also some cases in the file where this fixture is explicitly
    # overridden by a more local mark.  That is deliberate.
    return request.param


@pytest.fixture(scope="function", params=[
    pytest.param(rand_super, id="super"),
    pytest.param(rand_super_bcsz, id="super_bcz")
])
def superoperator(request, dimension):
    return request.param(dimension)


class TestSuperopReps:
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """

    def test_SuperChoiSuper(self, superoperator):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """

        choi_matrix = to_choi(superoperator)
        test_supe = to_super(choi_matrix)

        # Assert both that the result is close to expected, and has the right
        # type.
        assert (test_supe - superoperator).norm() < tol
        assert choi_matrix.type == "super" and choi_matrix.superrep == "choi"
        assert test_supe.type == "super" and test_supe.superrep == "super"

    @pytest.mark.parametrize('dimension', [2, 4])
    def test_SuperChoiChiSuper(self, dimension):
        """
        Superoperator: Converting two-qubit superoperator through
        Choi and chi representations goes back to right superoperator.
        """
        superoperator = super_tensor(
            rand_super(dimension), rand_super(dimension),
        )

        choi_matrix = to_choi(superoperator)
        chi_matrix = to_chi(choi_matrix)
        test_supe = to_super(chi_matrix)

        # Assert both that the result is close to expected, and has the right
        # type.
        assert (test_supe - superoperator).norm() < tol
        assert choi_matrix.type == "super" and choi_matrix.superrep == "choi"
        assert chi_matrix.type == "super" and chi_matrix.superrep == "chi"
        assert test_supe.type == "super" and test_supe.superrep == "super"

    def test_ChoiKrausChoi(self, superoperator):
        """
        Superoperator: Convert superoperator to Choi matrix and back.
        """
        choi_matrix = to_choi(superoperator)
        kraus_ops = to_kraus(choi_matrix)
        test_choi = kraus_to_choi(kraus_ops)

        # Assert both that the result is close to expected, and has the right
        # type.
        assert (test_choi - choi_matrix).norm() < tol
        assert choi_matrix.type == "super" and choi_matrix.superrep == "choi"
        assert test_choi.type == "super" and test_choi.superrep == "choi"

    def test_NonSquareKrausSuperChoi(self):
        """
        Superoperator: Convert non-square Kraus operator to Super + Choi matrix
        and back.
        """
        zero = asarray([[1], [0]], dtype=complex)
        one = asarray([[0], [1]], dtype=complex)
        zero_log = kron(kron(zero, zero), zero)
        one_log = kron(kron(one, one), one)
        # non-square Kraus operator (isometry)
        kraus = Qobj(zero_log @ zero.T + one_log @ one.T)
        super = sprepost(kraus, kraus.dag())
        choi = to_choi(super)
        op1 = to_kraus(super)
        op2 = to_kraus(choi)
        op3 = to_super(choi)

        assert choi.type == "super" and choi.superrep == "choi"
        assert super.type == "super" and super.superrep == "super"
        assert (op1[0] - kraus).norm() < tol
        assert (op2[0] - kraus).norm() < tol
        assert (op3 - super).norm() < tol

    def test_NeglectSmallKraus(self):
        """
        Superoperator: Convert Kraus to Choi matrix and back. Neglect tiny
        Kraus operators.
        """
        zero = asarray([[1], [0]], dtype=complex)
        one = asarray([[0], [1]], dtype=complex)
        zero_log = kron(kron(zero, zero), zero)
        one_log = kron(kron(one, one), one)
        # non-square Kraus operator (isometry)
        kraus = Qobj(zero_log @ zero.T + one_log @ one.T)
        super = sprepost(kraus, kraus.dag())
        # 1 non-zero Kraus operator the rest are zero
        sixteen_kraus_ops = to_kraus(super, tol=0.0)
        # default is tol=1e-9
        one_kraus_op = to_kraus(super)
        assert len(sixteen_kraus_ops) == 16 and len(one_kraus_op) == 1
        assert (one_kraus_op[0] - kraus).norm() < tol

    def test_SuperPreservesSelf(self, superoperator):
        """
        Superoperator: to_super(q) returns q if q is already a
        supermatrix.
        """

        assert superoperator is to_super(superoperator)

    def test_ChoiPreservesSelf(self, superoperator):
        """
        Superoperator: to_choi(q) returns q if q is already Choi.
        """
        choi = to_choi(superoperator)
        assert choi is to_choi(choi)

    def test_random_iscptp(self, superoperator):
        """
        Superoperator: Randomly generated superoperators are
        correctly reported as CPTP and HP.
        """
        assert superoperator.iscptp
        assert superoperator.ishp

    # Conjugation by a creation operator
    a = create(2).dag()
    S = sprepost(a, a.dag())

    # A single off-diagonal element
    S_ = sprepost(a, a)

    # Check that a linear combination of bipartite unitaries is CPTP and HP.
    S_U = (
        to_super(tensor(sigmax(), identity(2))) +
        to_super(tensor(identity(2), sigmay()))
    ) / 2

    # The partial transpose map, whose Choi matrix is SWAP
    ptr_swap = Qobj(swap(), type='super', superrep='choi')

    # Subnormalized maps (representing erasure channels, for instance)
    subnorm_map = Qobj(identity(4) * 0.9, type='super', superrep='super')

    @pytest.mark.parametrize(['qobj',  'shouldhp', 'shouldcp', 'shouldtp'], [
        pytest.param(S, True, True, False, id="conjugatio by create op"),
        pytest.param(S_, False, False, False, id="single off-diag"),
        pytest.param(identity(2), True, True, True, id="Identity"),
        pytest.param(sigmax(), True, True, True, id="Pauli X"),
        pytest.param(
            tensor(sigmax(), identity(2)), True, True, True,
            id="bipartite system",
        ),
        pytest.param(
            S_U, True, True, True, id="linear combination of bip. unitaries",
        ),
        pytest.param(ptr_swap,  True, False, True, id="partial transpose map"),
        pytest.param(subnorm_map, True, True, False, id="subnorm map"),
        pytest.param(basis(2), False, False, False, id="not an operator"),

    ])
    def test_known_iscptp(self, qobj, shouldhp, shouldcp, shouldtp):
        """
        Superoperator: ishp, iscp, istp and iscptp known cases.
        """
        assert qobj.ishp == shouldhp
        assert qobj.iscp == shouldcp
        assert qobj.istp == shouldtp
        assert qobj.iscptp == (shouldcp and shouldtp)

    def test_choi_tr(self, dimension):
        """
        Superoperator: Trace returned by to_choi matches docstring.
        """
        assert abs(to_choi(identity(dimension)).tr() - dimension) <= tol

    def test_stinespring_cp(self, dimension):
        """
        Stinespring: A and B match for CP maps.
        """
        superop = rand_super_bcsz(dimension)
        A, B = to_stinespring(superop)

        assert norm(A - B) < tol

    @pytest.mark.repeat(3)
    def test_stinespring_agrees(self, dimension):
        """
        Stinespring: Partial Tr over pair agrees w/ supermatrix.
        """

        map = rand_super_bcsz(dimension)
        state = rand_dm_ginibre(dimension)

        S = to_super(map)
        A, B = to_stinespring(map)

        q1 = vector_to_operator(
            S * operator_to_vector(state)
        )
        # FIXME: problem if Kraus index is implicitly
        #        ptraced!
        q2 = (A * state * B.dag()).ptrace((0,))

        assert (q1 - q2).norm('tr') <= tol

    def test_stinespring_dims(self, dimension):
        """
        Stinespring: Check that dims of channels are preserved.
        """
        chan = super_tensor(to_super(sigmax()), to_super(qeye(dimension)))
        A, B = to_stinespring(chan)
        assert A.dims == [[2, dimension, 1], [2, dimension]]
        assert B.dims == [[2, dimension, 1], [2, dimension]]

    @pytest.mark.parametrize('dimension', [2, 4, 8])
    def test_chi_choi_roundtrip(self, dimension):

        superop = rand_super_bcsz(dimension)
        superop = to_chi(superop)
        rt_superop = to_chi(to_choi(superop))
        dif = norm(rt_superop - superop)

        assert dif == pytest.approx(0, abs=1e-7)
        assert rt_superop.type == superop.type
        assert rt_superop.dims == superop.dims

    chi_sigmax = [
        [0, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    chi_diag2 = [
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    rotX_pi_4 = (-1j * sigmax() * pi / 4).expm()
    chi_rotX_pi_4 = [
        [2, 2j, 0, 0],
        [-2j, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    @pytest.mark.parametrize(['superop', 'chi_expected'], [
        pytest.param(sigmax(), chi_sigmax),
        pytest.param(to_super(sigmax()), chi_sigmax),
        pytest.param(qeye(2), chi_diag2),
        pytest.param(rotX_pi_4, chi_rotX_pi_4)
    ])
    def test_chi_known(self, superop, chi_expected):
        """
        Superoperator: Chi-matrix for known cases is correct.
        """
        chi_actual = to_chi(superop)
        chiq = Qobj(
            chi_expected, dims=[[[2], [2]], [[2], [2]]], superrep='chi',
        )
        assert (chi_actual - chiq).norm() < tol

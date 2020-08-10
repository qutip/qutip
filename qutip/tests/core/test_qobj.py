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

import numbers
import operator
import pytest

import numpy as np
import scipy.sparse
import scipy.linalg

import qutip


def _random_not_singular(N):
    """
    return a N*N complex array with determinant not 0.
    """
    data = np.zeros((1, 1))
    while np.linalg.det(data) == 0:
        data = np.random.random((N, N)) + \
               1j * np.random.random((N, N)) - (0.5 + 0.5j)
    return data


def assert_hermicity(oper, hermicity):
    # Check the cached isherm, if any exists.
    assert oper.isherm == hermicity
    # Force a reset of the cached value for isherm.
    oper._isherm = None
    # Force a recalculation of isherm.
    assert oper.isherm == hermicity


def test_QobjData():
    "qutip.Qobj data"
    N = 10
    data1 = _random_not_singular(N)
    q1 = qutip.Qobj(data1)
    assert isinstance(q1.data, qutip.core.data.Data)
    assert np.all(q1.data.to_array() == data1)

    data2 = _random_not_singular(N)
    data2 = scipy.sparse.csr_matrix(data2)
    q2 = qutip.Qobj(data2)
    assert isinstance(q2.data, qutip.core.data.Data)


def test_QobjType():
    "qutip.Qobj type"
    N = int(np.ceil(10.0 * np.random.random())) + 5

    ket_data = np.random.random((N, 1))
    ket_qobj = qutip.Qobj(ket_data)
    assert ket_qobj.type == 'ket'
    assert ket_qobj.isket

    bra_data = np.random.random((1, N))
    bra_qobj = qutip.Qobj(bra_data)
    assert bra_qobj.type == 'bra'
    assert bra_qobj.isbra

    oper_data = np.random.random((N, N))
    oper_qobj = qutip.Qobj(oper_data)
    assert oper_qobj.type == 'oper'
    assert oper_qobj.isoper

    N = 9
    super_data = np.random.random((N, N))
    super_qobj = qutip.Qobj(super_data, dims=[[[3]], [[3]]])
    assert super_qobj.type == 'super'
    assert super_qobj.issuper

    operket_qobj = qutip.operator_to_vector(oper_qobj)
    assert operket_qobj.isoperket
    assert operket_qobj.dag().isoperbra


class TestQobjHermicity:
    def test_standard(self):
        base = _random_not_singular(10)
        assert_hermicity(qutip.Qobj(base), False)
        assert_hermicity(qutip.Qobj(base + base.conj().T), True)
        assert_hermicity(qutip.destroy(5), False)
        assert_hermicity(qutip.create(5), False)

    def test_addition(self):
        q_a, q_ad = qutip.destroy(5), qutip.create(5)
        # test addition of two nonhermitian operators adding up to be hermitian
        q_x = q_a + q_ad
        assert_hermicity(q_x, True)
        # test addition of one hermitan and one nonhermitian operator
        assert_hermicity(q_x + q_a, False)
        # test addition of two hermitan operators
        assert_hermicity(q_x + q_x, True)

    def test_multiplication(self):
        # Test multiplication of two Hermitian operators.  This results in a
        # skew-Hermitian operator, so we're checking here that __mul__ doesn't
        # set wrong metadata.
        assert_hermicity(qutip.sigmax() * qutip.sigmay(), False)
        # Similarly, we need to check that -Z = X * iY is correctly identified
        # as Hermitian.
        assert_hermicity(1j * qutip.sigmax() * qutip.sigmay(), True)


def assert_unitarity(oper, unitarity):
    # Check the cached isunitary, if any exists.
    assert oper.isunitary == unitarity
    # Force a reset of the cached value for isunitary.
    oper._isunitary = None
    # Force a recalculation of isunitary.
    assert oper.isunitary == unitarity


def test_QobjUnitaryOper():
    "qutip.Qobj unitarity"
    # Check some standard operators
    Sx = qutip.sigmax()
    Sy = qutip.sigmay()
    assert_unitarity(qutip.qeye(4), True)
    assert_unitarity(Sx, True)
    assert_unitarity(Sy, True)
    assert_unitarity(qutip.sigmam(), False)
    assert_unitarity(qutip.destroy(10), False)
    # Check multiplcation of unitary is unitary
    assert_unitarity(Sx*Sy, True)
    # Check some other operations clear unitarity
    assert_unitarity(Sx+Sy, False)
    assert_unitarity(4*Sx, False)
    assert_unitarity(Sx*4, False)
    assert_unitarity(4+Sx, False)
    assert_unitarity(Sx+4, False)


def test_QobjDimsShape():
    "qutip.Qobj shape"
    N = 10
    data = _random_not_singular(N)

    q1 = qutip.Qobj(data)
    assert q1.dims == [[10], [10]]
    assert q1.shape == (10, 10)

    data = np.random.random((N, 1)) + 1j*np.random.random((N, 1)) - (0.5+0.5j)

    q1 = qutip.Qobj(data)
    assert q1.dims == [[10], [1]]
    assert q1.shape == (10, 1)

    data = _random_not_singular(4)

    q1 = qutip.Qobj(data, dims=[[2, 2], [2, 2]])
    assert q1.dims == [[2, 2], [2, 2]]
    assert q1.shape == (4, 4)


def test_QobjMulNonsquareDims():
    """
    qutip.Qobj: multiplication w/ non-square qobj.dims

    Checks for regression of #331.
    """
    data = np.array([[0, 1], [1, 0]])

    q1 = qutip.Qobj(data, dims=[[2, 1], [2]])
    q2 = qutip.Qobj(data, dims=[[2], [2]])

    assert (q1 * q2).dims == [[2, 1], [2]]
    assert (q2 * q1.dag()).dims == [[2], [2, 1]]
    assert (q1 * q2 * q1.dag()).dims == [[2, 1], [2, 1]]

    # Because of the above, we also need to check for extra indices
    # that aren't of length 1.
    q1 = qutip.Qobj([[1.+0.j,  0.+0.j],
                     [0.+0.j,  1.+0.j],
                     [0.+0.j,  1.+0.j],
                     [1.+0.j,  0.+0.j],
                     [0.+0.j,  0.-1.j],
                     [0.+1.j,  0.+0.j],
                     [1.+0.j,  0.+0.j],
                     [0.+0.j, -1.+0.j]],
                    dims=[[4, 2], [2]])
    assert (q1 * q2 * q1.dag()).dims == [[4, 2], [4, 2]]


def test_QobjAddition():
    "qutip.Qobj addition"
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])
    data3 = data1 + data2

    q1 = qutip.Qobj(data1)
    q2 = qutip.Qobj(data2)
    q3 = qutip.Qobj(data3)

    q4 = q1 + q2
    q4_type = q4.type
    q4_isherm = q4.isherm
    q4._type = None
    q4._isherm = None  # clear cached values
    assert q4_type == q4.type
    assert q4_isherm == q4.isherm

    # check elementwise addition/subtraction
    assert q3 == q4

    # check that addition is commutative
    assert q1 + q2 == q2 + q1

    data = np.random.random((5, 5))
    q = qutip.Qobj(data)
    x1 = q + 5
    x2 = 5 + q
    data = data + np.eye(5) * 5
    assert np.all(x1.full() == data)
    assert np.all(x2.full() == data)


def test_QobjSubtraction():
    "qutip.Qobj subtraction"
    data1 = _random_not_singular(5)
    q1 = qutip.Qobj(data1)

    data2 = _random_not_singular(5)
    q2 = qutip.Qobj(data2)

    q3 = q1 - q2
    data3 = data1 - data2
    assert np.all(q3.full() == data3)

    q4 = q2 - q1
    data4 = data2 - data1
    assert np.all(q4.full() == data4)


def test_QobjMultiplication():
    "qutip.Qobj multiplication"
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    data3 = np.dot(data1, data2)

    q1 = qutip.Qobj(data1)
    q2 = qutip.Qobj(data2)
    q3 = qutip.Qobj(data3)

    q4 = q1 * q2

    assert q3 == q4


def test_QobjDivision():
    "qutip.Qobj division"
    data = _random_not_singular(5)
    q = qutip.Qobj(data)
    randN = 10 * np.random.random()
    q = q / randN
    assert np.allclose(q.full(), data/randN)


def test_QobjPower():
    "qutip.Qobj power"
    data = _random_not_singular(5)
    q = qutip.Qobj(data)
    np.testing.assert_allclose((q**2).full(), data @ data, atol=1e-12)
    np.testing.assert_allclose((q**3).full(), data @ data @ data, atol=1e-12)


def test_QobjNeg():
    "qutip.Qobj negation"
    data = _random_not_singular(5)
    q = qutip.Qobj(data)
    x = -q
    assert np.all(x.full() == -data)
    assert q.isherm == x.isherm
    assert q.type == x.type


def test_QobjEquals():
    "qutip.Qobj equals"
    data = _random_not_singular(5)
    q1 = qutip.Qobj(data)
    q2 = qutip.Qobj(data)
    assert q1 == q2

    q1 = qutip.Qobj(data)
    q2 = qutip.Qobj(-data)
    assert q1 != q2


def test_QobjGetItem():
    "qutip.Qobj getitem"
    data = _random_not_singular(5)
    q = qutip.Qobj(data)
    assert q[0, 0] == data[0, 0]
    assert q[-1, 2] == data[-1, 2]


def test_CheckMulType():
    "qutip.Qobj multiplication type"
    # ket-bra and bra-ket multiplication
    psi = qutip.basis(5, 0)
    dm = psi * psi.dag()
    assert dm.isoper
    assert dm.isherm

    nrm = psi.dag() * psi
    assert isinstance(nrm, numbers.Complex)
    assert abs(nrm) == 1

    # operator-operator multiplication
    H1 = qutip.rand_herm(3)
    H2 = qutip.rand_herm(3)
    out = H1 * H2
    assert out.isoper
    out = H1 * H1
    assert out.isoper
    assert out.isherm
    out = H2 * H2
    assert out.isoper
    assert out.isherm

    U = qutip.rand_unitary(5)
    out = U.dag() * U
    assert out.isoper
    assert out.isherm

    N = qutip.num(5)

    out = N * N
    assert out.isoper
    assert out.isherm

    # operator-ket and bra-operator multiplication
    op = qutip.sigmax()
    ket1 = qutip.basis(2, 0)
    ket2 = op * ket1
    assert ket2.isket

    bra1 = qutip.basis(2, 0).dag()
    bra2 = bra1 * op
    assert bra2.isbra

    assert bra2.dag() == ket2

    # superoperator-operket and operbra-superoperator multiplication
    sop = qutip.to_super(qutip.sigmax())
    opket1 = qutip.operator_to_vector(qutip.fock_dm(2, 0))
    opket2 = sop * opket1
    assert opket2.isoperket

    opbra1 = qutip.operator_to_vector(qutip.fock_dm(2, 0)).dag()
    opbra2 = opbra1 * sop
    assert opbra2.isoperbra

    assert opbra2.dag() == opket2


def test_QobjConjugate():
    "qutip.Qobj conjugate"
    data = _random_not_singular(5)
    A = qutip.Qobj(data)
    B = A.conj()
    assert np.all(B.full() == data.conj())
    assert A.isherm == B.isherm
    assert A.type == B.type
    assert A.superrep == B.superrep


def test_QobjDagger():
    "qutip.Qobj adjoint (dagger)"
    data = _random_not_singular(5)
    A = qutip.Qobj(data)
    B = A.dag()
    assert np.all(B.full() == data.conj().T)
    assert A.isherm == B.isherm
    assert A.type == B.type
    assert A.superrep == B.superrep


def test_QobjDiagonals():
    "qutip.Qobj diagonals"
    data = _random_not_singular(5)
    A = qutip.Qobj(data)
    b = A.diag()
    assert np.all(b == np.diag(data))


def test_QobjEigenEnergies():
    "qutip.Qobj eigenenergies"
    data = np.eye(5)
    A = qutip.Qobj(data)
    b = A.eigenenergies()
    assert np.all(b == np.ones(5))

    data = np.diag(np.arange(10))
    A = qutip.Qobj(data)
    b = A.eigenenergies()
    assert np.all(b == np.arange(10))

    data = np.diag(np.arange(10))
    A = 5 * qutip.Qobj(data)
    b = A.eigenenergies()
    assert np.all(b == 5*np.arange(10))


def test_QobjEigenStates():
    "qutip.Qobj eigenstates"
    data = np.eye(5)
    A = qutip.Qobj(data)
    b, c = A.eigenstates()
    assert np.all(b == np.ones(5))
    kets = [qutip.basis(5, k) for k in range(5)]
    for k in range(5):
        assert c[k] == kets[k]


def test_QobjExpm():
    "qutip.Qobj expm (dense)"
    data = _random_not_singular(15)
    A = qutip.Qobj(data)
    B = A.expm()
    np.testing.assert_allclose(B.full(), scipy.linalg.expm(data), atol=1e-10)


def test_QobjExpmExplicitlySparse():
    "qutip.Qobj expm (sparse)"
    data = _random_not_singular(15)
    A = qutip.Qobj(data)
    B = A.expm(method='sparse')
    np.testing.assert_allclose(B.full(), scipy.linalg.expm(data), atol=1e-10)


def test_QobjExpmZeroOper():
    "qutip.Qobj expm zero_oper (#493)"
    A = qutip.Qobj(np.zeros((5, 5), dtype=complex))
    B = A.expm()
    assert B == qutip.qeye(5)


def test_Qobj_sqrtm():
    "qutip.Qobj sqrtm"
    data = _random_not_singular(5)
    A = qutip.Qobj(data)
    B = A.sqrtm()
    assert A == B * B


def test_Qobj_inv():
    "qutip.Qobj inv"
    data = _random_not_singular(5)
    A = qutip.Qobj(data)
    B = A.inv()
    assert qutip.qeye(5) == A * B
    assert qutip.qeye(5) == B * A
    B = A.inv(sparse=True)
    assert qutip.qeye(5) == A * B
    assert qutip.qeye(5) == B * A


def test_QobjFull():
    "qutip.Qobj full"
    data = _random_not_singular(15)
    A = qutip.Qobj(data)
    b = A.full()
    assert np.all(b == data)


def test_QobjNorm():
    "qutip.Qobj norm"
    # vector L2-norm test
    N = 20
    x = np.random.random(N) + 1j*np.random.random(N)
    A = qutip.Qobj(x)
    np.testing.assert_allclose(A.norm(), scipy.linalg.norm(x, 2), atol=1e-12)
    # vector max (inf) norm test
    np.testing.assert_allclose(A.norm('max'), scipy.linalg.norm(x, np.inf),
                               atol=1e-12)
    # operator frobius norm
    x = np.random.random((N, N)) + 1j * np.random.random((N, N))
    A = qutip.Qobj(x)
    np.testing.assert_allclose(A.norm('fro'), scipy.linalg.norm(x, 'fro'),
                               atol=1e-12)
    # operator trace norm
    a = qutip.rand_herm(10, 0.25)
    np.testing.assert_allclose(a.norm(), (a*a.dag()).sqrtm().tr().real)
    b = qutip.rand_herm(10, 0.25) - 1j*qutip.rand_herm(10, 0.25)
    np.testing.assert_allclose(b.norm(), (b*b.dag()).sqrtm().tr().real)


def test_QobjPurity():
    "Tests the purity method of `qutip.Qobj`"
    psi = qutip.basis(2, 1)
    # check purity of pure ket state
    np.testing.assert_allclose(psi.purity(), 1)
    # check purity of pure ket state (superposition)
    psi2 = qutip.basis(2, 0)
    psi_tot = (psi+psi2).unit()
    np.testing.assert_allclose(psi_tot.purity(), 1)
    # check purity of density matrix of pure state
    np.testing.assert_allclose(qutip.ket2dm(psi_tot).purity(), 1)
    # check purity of maximally mixed density matrix
    rho_mixed = (qutip.ket2dm(psi) + qutip.ket2dm(psi2)).unit()
    np.testing.assert_allclose(rho_mixed.purity(), 0.5)


def test_QobjPermute():
    "qutip.Qobj permute"
    A = qutip.basis(3, 0)
    B = qutip.basis(5, 4)
    C = qutip.basis(4, 2)
    psi = qutip.tensor(A, B, C)
    psi2 = psi.permute([2, 0, 1])
    assert psi2 == qutip.tensor(C, A, B)

    psi_bra = psi.dag()
    psi2_bra = psi_bra.permute([2, 0, 1])
    assert psi2_bra == qutip.tensor(C, A, B).dag()

    A = qutip.fock_dm(3, 0)
    B = qutip.fock_dm(5, 4)
    C = qutip.fock_dm(4, 2)
    rho = qutip.tensor(A, B, C)
    rho2 = rho.permute([2, 0, 1])
    assert rho2 == qutip.tensor(C, A, B)

    for _ in range(3):
        A = qutip.rand_ket(3)
        B = qutip.rand_ket(4)
        C = qutip.rand_ket(5)
        psi = qutip.tensor(A, B, C)
        psi2 = psi.permute([1, 0, 2])
        assert psi2 == qutip.tensor(B, A, C)

        psi_bra = psi.dag()
        psi2_bra = psi_bra.permute([1, 0, 2])
        assert psi2_bra == qutip.tensor(B, A, C).dag()

    for _ in range(3):
        A = qutip.rand_dm(3)
        B = qutip.rand_dm(4)
        C = qutip.rand_dm(5)
        rho = qutip.tensor(A, B, C)
        rho2 = rho.permute([1, 0, 2])
        assert rho2 == qutip.tensor(B, A, C)

        rho_vec = qutip.operator_to_vector(rho)
        rho2_vec = rho_vec.permute([[1, 0, 2], [4, 3, 5]])
        assert rho2_vec == qutip.operator_to_vector(qutip.tensor(B, A, C))

        rho_vec_bra = qutip.operator_to_vector(rho).dag()
        rho2_vec_bra = rho_vec_bra.permute([[1, 0, 2], [4, 3, 5]])
        assert (rho2_vec_bra
                == qutip.operator_to_vector(qutip.tensor(B, A, C)).dag())

    for _ in range(3):
        super_dims = [3, 5, 4]
        U = qutip.rand_unitary(np.prod(super_dims), density=0.02,
                               dims=[super_dims, super_dims])
        Unew = U.permute([2, 1, 0])
        S_tens = qutip.to_super(U)
        S_tens_new = qutip.to_super(Unew)
        assert S_tens_new == S_tens.permute([[2, 1, 0], [5, 4, 3]])


def test_KetType():
    "qutip.Qobj ket type"

    psi = qutip.basis(2, 1)

    assert psi.isket
    assert not psi.isbra
    assert not psi.isoper
    assert not psi.issuper

    psi = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 0))

    assert psi.isket
    assert not psi.isbra
    assert not psi.isoper
    assert not psi.issuper


def test_BraType():
    "qutip.Qobj bra type"

    psi = qutip.basis(2, 1).dag()
    assert not psi.isket
    assert psi.isbra
    assert not psi.isoper
    assert not psi.issuper

    psi = qutip.tensor(qutip.basis(2, 1).dag(), qutip.basis(2, 0).dag())
    assert not psi.isket
    assert psi.isbra
    assert not psi.isoper
    assert not psi.issuper


def test_OperType():
    "qutip.Qobj operator type"

    psi = qutip.basis(2, 1)
    rho = psi * psi.dag()

    assert not rho.isket
    assert not rho.isbra
    assert rho.isoper
    assert not rho.issuper


def test_SuperType():
    "qutip.Qobj superoperator type"

    psi = qutip.basis(2, 1)
    rho = psi * psi.dag()

    sop = qutip.spre(rho)

    assert not sop.isket
    assert not sop.isbra
    assert not sop.isoper
    assert sop.issuper

    sop = qutip.spost(rho)

    assert not sop.isket
    assert not sop.isbra
    assert not sop.isoper
    assert sop.issuper


@pytest.mark.parametrize("dimension", [2, 4, 8])
@pytest.mark.parametrize("conversion", [
    pytest.param(qutip.to_super, id='to_super'),
    pytest.param(qutip.to_choi, id='to_choi'),
    pytest.param(qutip.to_chi, id='to_chi'),
])
def test_dag_preserves_superrep(dimension, conversion):
    """
    Checks that dag() preserves superrep.
    """
    qobj = conversion(qutip.rand_super_bcsz(dimension))
    assert qobj.superrep == qobj.dag().superrep


@pytest.mark.parametrize("superrep", ["super", "choi", "chi"])
@pytest.mark.parametrize(["operation", "check_op", "check_scalar"], [
    pytest.param(operator.add, True, True, id='add'),
    pytest.param(operator.sub, True, True, id='sub'),
    pytest.param(operator.mul, True, True, id='mul'),
    pytest.param(operator.truediv, False, True, id='div'),
    pytest.param(qutip.tensor, True, False, id='tensor'),
])
def test_arithmetic_preserves_superrep(superrep,
                                       operation, check_op, check_scalar):
    """
    Checks that binary ops preserve 'superrep'.

    .. note::

        The random superoperators are not chosen in a way that reflects the
        structure of that superrep, but are simply random matrices.
    """
    dims = [[[2], [2]], [[2], [2]]]
    shape = (4, 4)
    S1 = qutip.Qobj(np.random.random(shape), superrep=superrep, dims=dims)
    S2 = qutip.Qobj(np.random.random(shape), superrep=superrep, dims=dims)
    x = np.random.random()

    check_list = []
    if check_op:
        check_list.append(operation(S1, S2))
    if check_scalar:
        check_list.append(operation(S1, x))
    if check_op and check_scalar:
        check_list.append(operation(x, S2))

    for S in check_list:
        assert S.issuper
        assert S.type == "super"
        assert S.superrep == superrep


def test_isherm_skew():
    """
    mul and tensor of skew-Hermitian operators report ``isherm = True``.
    """
    iH = 1j * qutip.rand_herm(5)
    assert_hermicity(iH, False)
    assert_hermicity(iH * iH, True)
    assert_hermicity(qutip.tensor(iH, iH), True)


def test_super_tensor_operket():
    """
    Tensor: Checks that super_tensor respects states.
    """
    rho1, rho2 = qutip.rand_dm(5), qutip.rand_dm(7)
    qutip.operator_to_vector(rho1)
    qutip.operator_to_vector(rho2)


def test_super_tensor_property():
    """
    Tensor: Super_tensor correctly tensors on underlying spaces.
    """
    U1 = qutip.rand_unitary(3)
    U2 = qutip.rand_unitary(5)
    U = qutip.tensor(U1, U2)
    S_tens = qutip.to_super(U)
    S_supertens = qutip.super_tensor(qutip.to_super(U1), qutip.to_super(U2))
    assert S_tens == S_supertens
    assert S_supertens.superrep == 'super'


def test_composite_oper():
    """
    Composite: Tests compositing unitaries and superoperators.
    """
    U1 = qutip.rand_unitary(3)
    U2 = qutip.rand_unitary(5)
    S1 = qutip.to_super(U1)
    S2 = qutip.to_super(U2)
    S3 = qutip.rand_super(4)
    S4 = qutip.rand_super(7)

    assert qutip.composite(U1, U2) == qutip.tensor(U1, U2)
    assert qutip.composite(S3, S4) == qutip.super_tensor(S3, S4)
    assert qutip.composite(U1, S4) == qutip.super_tensor(S1, S4)
    assert qutip.composite(S3, U2) == qutip.super_tensor(S3, S2)


def test_composite_vec():
    """
    Composite: Tests compositing states and density operators.
    """
    k1 = qutip.rand_ket(5)
    k2 = qutip.rand_ket(7)
    r1 = qutip.operator_to_vector(qutip.ket2dm(k1))
    r2 = qutip.operator_to_vector(qutip.ket2dm(k2))

    r3 = qutip.operator_to_vector(qutip.rand_dm(3))
    r4 = qutip.operator_to_vector(qutip.rand_dm(4))

    assert qutip.composite(k1, k2) == qutip.tensor(k1, k2)
    assert qutip.composite(r3, r4) == qutip.super_tensor(r3, r4)
    assert qutip.composite(k1, r4) == qutip.super_tensor(r1, r4)
    assert qutip.composite(r3, k2) == qutip.super_tensor(r3, r2)

# TODO: move out to a more appropriate module.


def trunc_neg_case(qobj, method, expected=None):
    pos_qobj = qobj.trunc_neg(method=method)
    assert all(energy > -1e-8 for energy in pos_qobj.eigenenergies())
    np.testing.assert_allclose(pos_qobj.tr(), 1)
    if expected is not None:
        test_array = pos_qobj.full()
        exp_array = expected.full()
        np.testing.assert_allclose(test_array, exp_array)


class TestTruncNeg:
    """Test qutip.Qobj.trunc_neg for several different cases."""
    def test_positive_operator(self):
        trunc_neg_case(qutip.rand_dm(5), 'clip')
        trunc_neg_case(qutip.rand_dm(5), 'sgs')

    def test_diagonal_operator(self):
        to_test = qutip.Qobj(np.diag([1.1, 0, -0.1]))
        expected = qutip.Qobj(np.diag([1.0, 0.0, 0.0]))
        trunc_neg_case(to_test, 'clip', expected)
        trunc_neg_case(to_test, 'sgs', expected)

    def test_nondiagonal_operator(self):
        U = qutip.rand_unitary(3)
        to_test = U * qutip.Qobj(np.diag([1.1, 0, -0.1])) * U.dag()
        expected = U * qutip.Qobj(np.diag([1.0, 0.0, 0.0])) * U.dag()
        trunc_neg_case(to_test, 'clip', expected)
        trunc_neg_case(to_test, 'sgs', expected)

    def test_sgs_known_good(self):
        trunc_neg_case(qutip.Qobj(np.diag([3./5, 1./2, 7./20, 1./10, -11./20])),
                       'sgs',
                       qutip.Qobj(np.diag([9./20, 7./20, 1./5, 0, 0])))


def test_cosm():
    """
    Test qutip.Qobj: cosm
    """
    A = qutip.rand_herm(5)
    B = A.cosm().full()
    C = scipy.linalg.cosm(A.full())
    np.testing.assert_allclose(B, C, atol=1e-14)


def test_sinm():
    """
    Test qutip.Qobj: sinm
    """
    A = qutip.rand_herm(5)
    B = A.sinm().full()
    C = scipy.linalg.sinm(A.full())
    np.testing.assert_allclose(B, C, atol=1e-14)


@pytest.mark.parametrize("sub_dimensions", ([2], [2, 2], [2, 3], [3, 5, 2]))
def test_dual_channel(sub_dimensions, n_trials=50):
    """
    qutip.Qobj: dual_chan() preserves inner products with arbitrary density ops.
    """
    S = qutip.rand_super_bcsz(np.prod(sub_dimensions))
    S.dims = [[sub_dimensions, sub_dimensions],
              [sub_dimensions, sub_dimensions]]
    S = qutip.to_super(S)
    left_dims, right_dims = S.dims

    # Assume for the purposes of the test that S maps square operators to
    # square operators.
    in_dim = np.prod(right_dims[0])
    out_dim = np.prod(left_dims[0])

    S_dual = qutip.to_super(S.dual_chan())

    primals = []
    duals = []

    for _ in [None]*n_trials:
        X = qutip.rand_dm_ginibre(out_dim)
        X.dims = left_dims
        X = qutip.operator_to_vector(X)
        Y = qutip.rand_dm_ginibre(in_dim)
        Y.dims = right_dims
        Y = qutip.operator_to_vector(Y)

        primals.append(X.dag() * S * Y)
        duals.append(X.dag() * S_dual.dag() * Y)

    np.testing.assert_allclose(primals, duals)


def test_call():
    """
    Test qutip.Qobj: Call
    """
    # Make test objects.
    psi = qutip.rand_ket(3)
    rho = qutip.rand_dm_ginibre(3)
    U = qutip.rand_unitary(3)
    S = qutip.rand_super_bcsz(3)

    # Case 0: oper(ket).
    assert U(psi) == U * psi
    # Case 1: oper(oper). Should raise TypeError.
    with pytest.raises(TypeError):
        U(rho)
    # Case 2: super(ket).
    expected = qutip.vector_to_operator(S*qutip.operator_to_vector(psi.proj()))
    assert S(psi) == expected
    # Case 3: super(oper).
    expected = qutip.vector_to_operator(S * qutip.operator_to_vector(rho))
    assert S(rho) == expected
    # Case 4: super(super). Should raise TypeError.
    with pytest.raises(TypeError):
        S(S)


def test_mat_elem():
    """
    Test qutip.Qobj: Compute matrix elements
    """
    for _ in range(10):
        N = 20
        H = qutip.rand_herm(N, 0.2)
        L = qutip.rand_ket(N, 0.3)
        Ld = L.dag()
        R = qutip.rand_ket(N, 0.3)
        ans = Ld * H * R
        # bra-ket
        out1 = H.matrix_element(Ld, R)
        # ket-ket
        out2 = H.matrix_element(Ld, R)
        assert abs(ans - out1) < 1e-14
        assert abs(ans - out2) < 1e-14


def test_projection():
    """
    Test qutip.Qobj: Projection operator
    """
    for _ in range(10):
        N = 5
        K = qutip.tensor(qutip.rand_ket(N, 0.75), qutip.rand_ket(N, 0.75))
        B = K.dag()
        ans = K * K.dag()
        out1 = K.proj()
        out2 = B.proj()
        assert out1 == ans
        assert out2 == ans


def test_overlap():
    """
    Test qutip.Qobj: Overlap (inner product)
    """
    for _ in range(10):
        N = 10
        A = qutip.rand_ket(N, 0.75)
        Ad = A.dag()
        B = qutip.rand_ket(N, 0.75)
        Bd = B.dag()
        ans = A.dag() * B
        np.testing.assert_allclose(A.overlap(B), ans)
        np.testing.assert_allclose(Ad.overlap(B), ans)
        np.testing.assert_allclose(Ad.overlap(Bd), ans)
        np.testing.assert_allclose(A.overlap(Bd), np.conj(ans))


def test_unit():
    """
    Test qutip.Qobj: unit
    """
    psi = (10*np.random.randn()*qutip.basis(2, 0)
           - 10j*np.random.randn()*qutip.basis(2, 1))
    psi2 = psi.unit()
    psi.unit(inplace=True)
    assert psi == psi2
    np.testing.assert_allclose(np.linalg.norm(psi.full()), 1.0)


@pytest.mark.parametrize('inplace', [True, False], ids=['inplace', 'new'])
@pytest.mark.parametrize(['expanded', 'contracted'], [
    pytest.param([[1, 2, 2], [1, 2, 2]], [[2, 2], [2, 2]], id='op'),
    pytest.param([[2, 1, 1], [2, 1, 1]], [[2], [2]], id='op'),
    pytest.param([[5, 5], [5, 5]], [[5, 5], [5, 5]], id='op,unchanged'),
    pytest.param([[5], [5]], [[5], [5]], id='op,unchanged'),
    pytest.param([[5, 1, 3, 1], [5, 2, 3, 1]], [[5, 1, 3], [5, 2, 3]],
                 id='mixed'),
    pytest.param([[2, 1, 2], [2, 2, 2]], [[2, 1, 2], [2, 2, 2]],
                 id='mixed,unchanged'),
    pytest.param([[2, 2, 2], [1, 2, 2]], [[2, 2, 2], [1, 2, 2]],
                 id='mixed,unchanged'),
    pytest.param([[2, 2, 1], [1, 1, 1]], [[2, 2], [1, 1]], id='ket'),
    pytest.param([[1, 2, 1], [1, 1, 1]], [[2], [1]], id='ket'),
    pytest.param([[2, 3, 4], [1, 1, 1]], [[2, 3, 4], [1, 1, 1]],
                 id='ket,unchanged'),
    pytest.param([[1, 1, 1], [2, 2, 1]], [[1, 1], [2, 2]], id='bra'),
    pytest.param([[1, 1, 1], [1, 2, 1]], [[1], [2]], id='bra'),
    pytest.param([[1, 1, 1], [2, 3, 4]], [[1, 1, 1], [2, 3, 4]],
                 id='bra,unchanged'),
    pytest.param([[[2, 1, 1], [2, 1, 1]], [1]], [[[2], [2]], [1]],
                 id='operket'),
    pytest.param([[1], [[2, 1, 1], [2, 1, 1]]], [[1], [[2], [2]]],
                 id='operbra'),
])
def test_contract(expanded, contracted, inplace):
    shape = (np.prod(contracted[0]), np.prod(contracted[1]))
    data = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    qobj = qutip.Qobj(data, dims=expanded)
    assert qobj.dims == expanded
    out = qobj.contract(inplace=inplace)
    if inplace:
        assert out is qobj
    else:
        assert out is not qobj
    assert out.dims == contracted
    assert out.shape == qobj.shape
    assert np.all(out.full() == qobj.full())

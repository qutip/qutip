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

import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy.testing import assert_equal, assert_, assert_almost_equal, run_module_suite

from qutip.qobj import Qobj
from qutip.random_objects import (rand_ket, rand_dm, rand_herm, rand_unitary,
                                  rand_super, rand_super_bcsz, rand_dm_ginibre)
from qutip.states import basis, fock_dm, ket2dm
from qutip.operators import create, destroy, num, sigmax, sigmay, qeye
from qutip.superoperator import spre, spost, operator_to_vector, vector_to_operator
from qutip.superop_reps import to_super, to_choi, to_chi
from qutip.tensor import tensor, super_tensor, composite

from operator import add, mul, truediv, sub
from functools import partial
from contextlib import contextmanager

@contextmanager
def expect_exception(ex_class):
    """
    Context manager that raises an AssertionError if its body
    does not raise a particular class of exception.
    """
    ex_name = getattr(ex_class, '__name__', str(ex_class))

    try:
        yield
    except ex_class:
        pass
    except Exception as other_ex:
        raise AssertionError(
            "Expected {}, but {} was raised.".format(
                ex_name, getattr(type(other_ex), '__name__', str(type(other_ex)))
            )
        )
    else:
        raise AssertionError("Expected {}, but nothing was raised.".format(ex_name))

def test_expect_exception():
    with expect_exception(ValueError):
        raise ValueError

    with expect_exception(AssertionError):
        with expect_exception(ValueError):
            pass

def assert_hermicity(oper, hermicity, msg=""):
    # Check the cached isherm, if any exists.
    assert_(oper.isherm == hermicity, msg)
    # Force a reset of the cached value for isherm.
    oper._isherm = None
    # Force a recalculation of isherm.
    assert_(oper.isherm == hermicity, msg)

def test_QobjData():
    "Qobj data"
    N = 10
    data1 = np.random.random(
        (N, N)) + 1j * np.random.random((N, N)) - (0.5 + 0.5j)
    q1 = Qobj(data1)
    # check if data is a csr_matrix if originally array
    assert_equal(sp.isspmatrix_csr(q1.data), True)
    # check if dense ouput is equal to original data
    assert_(np.all(q1.data.todense() - np.matrix(data1) == 0))

    data2 = np.random.random(
        (N, N)) + 1j * np.random.random((N, N)) - (0.5 + 0.5j)
    data2 = sp.csr_matrix(data2)
    q2 = Qobj(data2)
    # check if data is a csr_matrix if originally csr_matrix
    assert_equal(sp.isspmatrix_csr(q2.data), True)

    data3 = 1
    q3 = Qobj(data3)
    # check if data is a csr_matrix if originally int
    assert_equal(sp.isspmatrix_csr(q3.data), True)

    data4 = np.random.random(
        (N, N)) + 1j * np.random.random((N, N)) - (0.5 + 0.5j)
    data4 = np.matrix(data4)
    q4 = Qobj(data4)
    # check if data is a csr_matrix if originally csr_matrix
    assert_equal(sp.isspmatrix_csr(q4.data), True)
    assert_(np.all(q4.data.todense() - np.matrix(data4) == 0))


def test_QobjType():
    "Qobj type"
    N = int(np.ceil(10.0 * np.random.random())) + 5

    ket_data = np.random.random((N, 1))
    ket_qobj = Qobj(ket_data)
    assert_equal(ket_qobj.type, 'ket')
    assert_(ket_qobj.isket)

    bra_data = np.random.random((1, N))
    bra_qobj = Qobj(bra_data)
    assert_equal(bra_qobj.type, 'bra')
    assert_(bra_qobj.isbra)

    oper_data = np.random.random((N, N))
    oper_qobj = Qobj(oper_data)
    assert_equal(oper_qobj.type, 'oper')
    assert_(oper_qobj.isoper)

    N = 9
    super_data = np.random.random((N, N))
    super_qobj = Qobj(super_data, dims=[[[3]], [[3]]])
    assert_equal(super_qobj.type, 'super')
    assert_(super_qobj.issuper)

    operket_qobj = operator_to_vector(oper_qobj)
    assert_(operket_qobj.isoperket)
    operbra_qobj = operket_qobj.dag()
    assert_(operbra_qobj.isoperbra)


def test_QobjHerm():
    "Qobj Hermicity"
    N = 10
    data = np.random.random(
        (N, N)) + 1j * np.random.random((N, N)) - (0.5 + 0.5j)
    q = Qobj(data)
    assert_equal(q.isherm, False)

    data = data + data.conj().T
    q = Qobj(data)
    assert_(q.isherm)

    q_a = destroy(5)
    assert_(not q_a.isherm)

    q_ad = create(5)
    assert_(not q_ad.isherm)

    # test addition of two nonhermitian operators adding up to a hermitian one
    q_x = q_a + q_ad
    assert_hermicity(q_x, True)

    # test addition of one hermitan and one nonhermitian operator
    q = q_x + q_a
    assert_hermicity(q, False)

    # test addition of two hermitan operators
    q = q_x + q_x
    assert_hermicity(q, True)

    # Test multiplication of two Hermitian operators.
    # This results in a skew-Hermitian operator, so
    # we're checking here that __mul__ doesn't set wrong
    # metadata.
    q = sigmax() * sigmay()
    assert_hermicity(q, False, "Expected iZ = X * Y to be skew-Hermitian.")
    # Similarly, we need to check that -Z = X * iY is correctly
    # identified as Hermitian.
    q = sigmax() * (1j * sigmay())
    assert_hermicity(q, True, "Expected -Z = X * iY to be Hermitian.")



def test_QobjDimsShape():
    "Qobj shape"
    N = 10
    data = np.random.random(
        (N, N)) + 1j * np.random.random((N, N)) - (0.5 + 0.5j)

    q1 = Qobj(data)
    assert_equal(q1.dims, [[10], [10]])
    assert_equal(q1.shape, [10, 10])

    data = np.random.random(
        (N, 1)) + 1j * np.random.random((N, 1)) - (0.5 + 0.5j)

    q1 = Qobj(data)
    assert_equal(q1.dims, [[10], [1]])
    assert_equal(q1.shape, [10, 1])

    N = 4

    data = np.random.random(
        (N, N)) + 1j * np.random.random((N, N)) - (0.5 + 0.5j)

    q1 = Qobj(data, dims=[[2, 2], [2, 2]])
    assert_equal(q1.dims, [[2, 2], [2, 2]])
    assert_equal(q1.shape, [4, 4])

def test_QobjMulNonsquareDims():
    """
    Qobj: multiplication w/ non-square qobj.dims

    Checks for regression of #331.
    """
    data = np.array([[0, 1], [1, 0]])

    q1 = Qobj(data)
    q1.dims[0].append(1)
    q2 = Qobj(data)

    assert_equal((q1 * q2).dims, [[2, 1], [2]])
    assert_equal((q2 * q1.dag()).dims, [[2], [2, 1]])

    # Note that this is [[2], [2]] instead of [[2, 1], [2, 1]],
    # as matching dimensions of 1 are implicitly partial traced out.
    # (See #331.)
    assert_equal((q1 * q2 * q1.dag()).dims, [[2], [2]])
    
    # Because of the above, we also need to check for extra indices
    # that aren't of length 1.
    q1 = Qobj([[ 1.+0.j,  0.+0.j],
         [ 0.+0.j,  1.+0.j],
         [ 0.+0.j,  1.+0.j],
         [ 1.+0.j,  0.+0.j],
         [ 0.+0.j,  0.-1.j],
         [ 0.+1.j,  0.+0.j],
         [ 1.+0.j,  0.+0.j],
         [ 0.+0.j, -1.+0.j]
     ], dims=[[4, 2], [2]])

    assert_equal((q1 * q2 * q1.dag()).dims, [[4, 2], [4, 2]])

def test_QobjAddition():
    "Qobj addition"
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    data3 = data1 + data2

    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)

    q4 = q1 + q2

    q4_type = q4.type
    q4_isherm = q4.isherm
    q4._type = None
    q4._isherm = None  # clear cached values
    assert_equal(q4_type, q4.type)
    assert_equal(q4_isherm, q4.isherm)

    # check elementwise addition/subtraction
    assert_(q3 == q4)

    # check that addition is commutative
    assert_(q1 + q2 == q2 + q1)

    data = np.random.random((5, 5))
    q = Qobj(data)

    x1 = q + 5
    x2 = 5 + q

    data = data + np.eye(5) * 5
    assert_(np.all(x1.data.todense() - np.matrix(data) == 0))
    assert_(np.all(x2.data.todense() - np.matrix(data) == 0))

    data = np.random.random((5, 5))
    q = Qobj(data)
    x3 = q + data
    x4 = data + q

    data = 2.0 * data
    assert_(np.all(x3.data.todense() - np.matrix(data) == 0))
    assert_(np.all(x4.data.todense() - np.matrix(data) == 0))


def test_QobjSubtraction():
    "Qobj subtraction"
    data1 = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q1 = Qobj(data1)

    data2 = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q2 = Qobj(data2)

    q3 = q1 - q2
    data3 = data1 - data2

    assert_(np.all(q3.data.todense() - np.matrix(data3) == 0))

    q4 = q2 - q1
    data4 = data2 - data1

    assert_(np.all(q4.data.todense() - np.matrix(data4) == 0))


def test_QobjMultiplication():
    "Qobj multiplication"
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    data3 = np.dot(data1, data2)

    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)

    q4 = q1 * q2

    assert_(q3 == q4)


def test_QobjDivision():
    "Qobj division"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q = Qobj(data)
    randN = 10 * np.random.random()
    q = q / randN
    assert_(np.allclose(q.data.todense(), np.matrix(data) / randN))


def test_QobjPower():
    "Qobj power"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q = Qobj(data)

    q2 = q ** 2
    assert_((q2.data.todense() - np.matrix(data) ** 2 < 1e-12).all())

    q3 = q ** 3
    assert_((q3.data.todense() - np.matrix(data) ** 3 < 1e-12).all())


def test_QobjNeg():
    "Qobj negation"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q = Qobj(data)
    x = -q
    assert_(np.all(x.data.todense() + np.matrix(data) == 0))
    assert_equal(q.isherm, x.isherm)
    assert_equal(q.type, x.type)


def test_QobjEquals():
    "Qobj equals"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q1 = Qobj(data)
    q2 = Qobj(data)
    assert_(q1 == q2)

    q1 = Qobj(data)
    q2 = Qobj(-data)
    assert_(q1 != q2)


def test_QobjGetItem():
    "Qobj getitem"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    q = Qobj(data)
    assert_equal(q[0, 0], data[0, 0])
    assert_equal(q[-1, 2], data[-1, 2])


def test_CheckMulType():
    "Qobj multiplication type"

    # ket-bra and bra-ket multiplication
    psi = basis(5)
    dm = psi * psi.dag()
    assert_(dm.isoper)
    assert_(dm.isherm)

    nrm = psi.dag() * psi
    assert_equal(np.prod(nrm.shape), 1)
    assert_((abs(nrm) == 1)[0, 0])

    # operator-operator multiplication
    H1 = rand_herm(3)
    H2 = rand_herm(3)
    out = H1 * H2
    assert_(out.isoper)
    out = H1 * H1
    assert_(out.isoper)
    assert_(out.isherm)
    out = H2 * H2
    assert_(out.isoper)
    assert_(out.isherm)

    U = rand_unitary(5)
    out = U.dag() * U
    assert_(out.isoper)
    assert_(out.isherm)

    N = num(5)

    out = N * N
    assert_(out.isoper)
    assert_(out.isherm)

    # operator-ket and bra-operator multiplication
    op = sigmax()
    ket1 = basis(2)
    ket2 = op * ket1
    assert_(ket2.isket)

    bra1 = basis(2).dag()
    bra2 = bra1 * op
    assert_(bra2.isbra)

    assert_(bra2.dag() == ket2)

    # superoperator-operket and operbra-superoperator multiplication
    sop = to_super(sigmax())
    opket1 = operator_to_vector(fock_dm(2))
    opket2 = sop * opket1
    assert(opket2.isoperket)

    opbra1 = operator_to_vector(fock_dm(2)).dag()
    opbra2 = opbra1 * sop
    assert(opbra2.isoperbra)

    assert_(opbra2.dag() == opket2)

def test_Qobj_Spmv():
    "Qobj mult ndarray right"
    A = rand_herm(5)
    b = rand_ket(5).full()
    C = A*b
    D = A.full().dot(b)
    assert_(np.all((C-D)<1e-14))


def test_QobjConjugate():
    "Qobj conjugate"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    A = Qobj(data)
    B = A.conj()
    assert_(np.all(B.data.todense() - np.matrix(data.conj()) == 0))
    assert_equal(A.isherm, B.isherm)
    assert_equal(A.type, B.type)
    assert_equal(A.superrep, B.superrep)


def test_QobjDagger():
    "Qobj adjoint (dagger)"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    A = Qobj(data)
    B = A.dag()
    assert_(np.all(B.data.todense() - np.matrix(data.conj().T) == 0))
    assert_equal(A.isherm, B.isherm)
    assert_equal(A.type, B.type)
    assert_equal(A.superrep, B.superrep)


def test_QobjDiagonals():
    "Qobj diagonals"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    A = Qobj(data)
    b = A.diag()
    assert_(np.all(b - np.diag(data) == 0))


def test_QobjEigenEnergies():
    "Qobj eigenenergies"
    data = np.eye(5)
    A = Qobj(data)
    b = A.eigenenergies()
    assert_(np.all(b - np.ones(5) == 0))

    data = np.diag(np.arange(10))
    A = Qobj(data)
    b = A.eigenenergies()
    assert_(np.all(b - np.arange(10) == 0))

    data = np.diag(np.arange(10))
    A = 5 * Qobj(data)
    b = A.eigenenergies()
    assert_(np.all(b - 5 * np.arange(10) == 0))


def test_QobjEigenStates():
    "Qobj eigenstates"
    data = np.eye(5)
    A = Qobj(data)
    b, c = A.eigenstates()
    assert_(np.all(b - np.ones(5) == 0))

    kets = [basis(5, k) for k in range(5)]

    for k in range(5):
        assert_(c[k] == kets[k])


def test_QobjExpm():
    "Qobj expm (dense)"
    data = np.random.random(
        (15, 15)) + 1j * np.random.random((15, 15)) - (0.5 + 0.5j)
    A = Qobj(data)
    B = A.expm()
    assert_((B.data.todense() - np.matrix(la.expm(data)) < 1e-10).all())


def test_QobjExpmExplicitlySparse():
    "Qobj expm (sparse)"
    data = np.random.random(
        (15, 15)) + 1j * np.random.random((15, 15)) - (0.5 + 0.5j)
    A = Qobj(data)
    B = A.expm(method='sparse')
    assert_((B.data.todense() - np.matrix(la.expm(data)) < 1e-10).all())


def test_QobjExpmZeroOper():
    "Qobj expm zero_oper (#493)"
    A = Qobj(np.zeros((5,5), dtype=complex))
    B = A.expm()
    assert_(B == qeye(5))


def test_Qobj_sqrtm():
    "Qobj sqrtm"
    data = np.random.random(
        (5, 5)) + 1j * np.random.random((5, 5)) - (0.5 + 0.5j)
    A = Qobj(data)
    B = A.sqrtm()
    assert_(A == B * B)


def test_QobjFull():
    "Qobj full"
    data = np.random.random(
        (15, 15)) + 1j * np.random.random((15, 15)) - (0.5 + 0.5j)
    A = Qobj(data)
    b = A.full()
    assert_(np.all(b - data == 0))


def test_QobjNorm():
    "Qobj norm"
    # vector L2-norm test
    N = 20
    x = np.random.random(N) + 1j * np.random.random(N)
    A = Qobj(x)
    assert_equal(np.abs(A.norm() - la.norm(A.data.data, 2)) < 1e-12, True)
    # vector max (inf) norm test
    assert_equal(
        np.abs(A.norm('max') - la.norm(A.data.data, np.inf)) < 1e-12, True)
    # operator frobius norm
    x = np.random.random((N, N)) + 1j * np.random.random((N, N))
    A = Qobj(x)
    assert_equal(
        np.abs(A.norm('fro') - la.norm(A.full(), 'fro')) < 1e-12, True)


def test_QobjPermute():
    "Qobj permute"
    A = basis(5, 0)
    B = basis(5, 4)
    C = basis(5, 2)
    psi = tensor(A, B, C)
    psi2 = psi.permute([2, 0, 1])
    assert_(psi2 == tensor(C, A, B))
    
    psi_bra = psi.dag()
    psi2_bra = psi_bra.permute([2, 0, 1])
    assert_(psi2_bra == tensor(C, A, B).dag())

    A = fock_dm(5, 0)
    B = fock_dm(5, 4)
    C = fock_dm(5, 2)
    rho = tensor(A, B, C)
    rho2 = rho.permute([2, 0, 1])
    assert_(rho2 == tensor(C, A, B))

    for ii in range(3):
        A = rand_ket(5)
        B = rand_ket(5)
        C = rand_ket(5)
        psi = tensor(A, B, C)
        psi2 = psi.permute([1, 0, 2])
        assert_(psi2 == tensor(B, A, C))
        
        psi_bra = psi.dag()
        psi2_bra = psi_bra.permute([1, 0, 2])
        assert_(psi2_bra == tensor(B, A, C).dag())

    for ii in range(3):
        A = rand_dm(5)
        B = rand_dm(5)
        C = rand_dm(5)
        rho = tensor(A, B, C)
        rho2 = rho.permute([1, 0, 2])
        assert_(rho2 == tensor(B, A, C))


def test_KetType():
    "Qobj ket type"

    psi = basis(2, 1)

    assert_(psi.isket)
    assert_(not psi.isbra)
    assert_(not psi.isoper)
    assert_(not psi.issuper)

    psi = tensor(basis(2, 1), basis(2, 0))

    assert_(psi.isket)
    assert_(not psi.isbra)
    assert_(not psi.isoper)
    assert_(not psi.issuper)


def test_BraType():
    "Qobj bra type"

    psi = basis(2, 1).dag()

    assert_equal(psi.isket, False)
    assert_equal(psi.isbra, True)
    assert_equal(psi.isoper, False)
    assert_equal(psi.issuper, False)

    psi = tensor(basis(2, 1).dag(), basis(2, 0).dag())

    assert_equal(psi.isket, False)
    assert_equal(psi.isbra, True)
    assert_equal(psi.isoper, False)
    assert_equal(psi.issuper, False)


def test_OperType():
    "Qobj operator type"

    psi = basis(2, 1)
    rho = psi * psi.dag()

    assert_equal(rho.isket, False)
    assert_equal(rho.isbra, False)
    assert_equal(rho.isoper, True)
    assert_equal(rho.issuper, False)


def test_SuperType():
    "Qobj superoperator type"

    psi = basis(2, 1)
    rho = psi * psi.dag()

    sop = spre(rho)

    assert_equal(sop.isket, False)
    assert_equal(sop.isbra, False)
    assert_equal(sop.isoper, False)
    assert_equal(sop.issuper, True)

    sop = spost(rho)

    assert_equal(sop.isket, False)
    assert_equal(sop.isbra, False)
    assert_equal(sop.isoper, False)
    assert_equal(sop.issuper, True)


def test_dag_preserves_superrep():
    """
    Checks that dag() preserves superrep.
    """

    def case(qobj):
        orig_superrep = qobj.superrep
        assert_equal(qobj.dag().superrep, orig_superrep)

    for dim in (2, 4, 8):
        qobj = rand_super_bcsz(dim)
        yield case, to_super(qobj)
        # These two shouldn't even do anything, since qobj
        # is Hermicity-preserving.
        yield case, to_choi(qobj)
        yield case, to_chi(qobj)


def test_arithmetic_preserves_superrep():
    """
    Checks that binary ops preserve 'superrep'.

    .. note::

        The random superoperators are not chosen in a way that reflects the
        structure of that superrep, but are simply random matrices.
    """

    dims = [[[2], [2]], [[2], [2]]]
    shape = (4, 4)

    def check(superrep, operation, chk_op, chk_scalar):
        S1 = Qobj(np.random.random(shape), superrep=superrep, dims=dims)
        S2 = Qobj(np.random.random(shape), superrep=superrep, dims=dims)
        x = np.random.random()

        check_list = []
        if chk_op:
            check_list.append(operation(S1, S2))
        if chk_scalar:
            check_list.append(operation(S1, x))
        if chk_op and chk_scalar:
            check_list.append(operation(x, S2))

        for S in check_list:
            assert_equal(S.type, "super",
                         "Operator {} did not preserve type='super'.".format(
                             operation)
                         )
            assert_equal(S.superrep, superrep,
                         "Operator {} did not preserve superrep={}.".format(
                             operation, superrep)
                         )

    for superrep in ['super', 'choi', 'chi']:
        for operation, chk_op, chk_scalar in [
                (add, True, True),
                (sub, True, True),
                (mul, True, True),
                (truediv, False, True),
                (tensor, True, False)
        ]:
            yield check, superrep, operation, chk_op, chk_scalar


def test_isherm_skew():
    """
    mul and tensor of skew-Hermitian operators report ``isherm = True``.
    """
    iH = 1j * rand_herm(5)

    assert_(not iH.isherm)
    assert_((iH * iH).isherm)
    assert_(tensor(iH, iH).isherm)


def test_super_tensor_operket():
    """
    Tensor: Checks that super_tensor respects states.
    """
    rho1, rho2 = rand_dm(5), rand_dm(7)
    operator_to_vector(rho1)
    operator_to_vector(rho2)


def test_super_tensor_property():
    """
    Tensor: Super_tensor correctly tensors on underlying spaces.
    """
    U1 = rand_unitary(3)
    U2 = rand_unitary(5)

    U = tensor(U1, U2)
    S_tens = to_super(U)

    S_supertens = super_tensor(to_super(U1), to_super(U2))

    assert_(S_tens == S_supertens)
    assert_equal(S_supertens.superrep, 'super')


def test_composite_oper():
    """
    Composite: Tests compositing unitaries and superoperators.
    """
    U1 = rand_unitary(3)
    U2 = rand_unitary(5)
    S1 = to_super(U1)
    S2 = to_super(U2)

    S3 = rand_super(4)
    S4 = rand_super(7)

    assert_(composite(U1, U2) == tensor(U1, U2))
    assert_(composite(S3, S4) == super_tensor(S3, S4))
    assert_(composite(U1, S4) == super_tensor(S1, S4))
    assert_(composite(S3, U2) == super_tensor(S3, S2))


def test_composite_vec():
    """
    Composite: Tests compositing states and density operators.
    """
    k1 = rand_ket(5)
    k2 = rand_ket(7)
    r1 = operator_to_vector(ket2dm(k1))
    r2 = operator_to_vector(ket2dm(k2))

    r3 = operator_to_vector(rand_dm(3))
    r4 = operator_to_vector(rand_dm(4))

    assert_(composite(k1, k2) == tensor(k1, k2))
    assert_(composite(r3, r4) == super_tensor(r3, r4))
    assert_(composite(k1, r4) == super_tensor(r1, r4))
    assert_(composite(r3, k2) == super_tensor(r3, r2))

# TODO: move out to a more appropriate module.

def has_description(case):
    """
    Decorates a test case such that it takes an argument describing
    that case, such that failure logs are usefully formatted.
    """
    # See https://code.google.com/archive/p/python-nose/issues/244#c1
    # for why this works.

    def make_case(description):
        partial_case = partial(case)
        partial_case.description = description
        return partial_case
    return make_case

def test_trunc_neg():
    """
    Test Qobj: Checks trunc_neg in several different cases.
    """

    @has_description
    def case(qobj, method, expected=None):
        pos_qobj = qobj.trunc_neg(method=method)
        assert(all([energy > -1e-8 for energy in pos_qobj.eigenenergies()]))
        assert_almost_equal(pos_qobj.tr(), 1)

        if expected is not None:
            assert_almost_equal(pos_qobj.data.todense(), expected.data.todense())

    for method in ('clip', 'sgs'):
        # Make sure that it works for operators that are already positive.
        yield case("Test Qobj: trunc_neg works for positive opers."), \
            rand_dm(5), method
        # Make sure that it works for a diagonal matrix.
        yield case("Test Qobj: trunc_neg works for diagonal opers."), \
            Qobj(np.diag([1.1, -0.1])), method, Qobj(np.diag([1.0, 0.0]))
        # Make sure that it works for a non-diagonal matrix.
        U = rand_unitary(3)
        yield case("Test Qobj: trunc_neg works for non-diagonal opers."), \
            U * Qobj(np.diag([1.1, 0, -0.1])) * U.dag(), \
            method, \
            U * Qobj(np.diag([1.0, 0.0, 0.0])) * U.dag()

    # Check the test case in SGS.
    yield (
        case("Test Qobj: trunc_neg works for SGS known-good test case."),
        Qobj(np.diag([3. / 5, 1. / 2, 7. / 20, 1. / 10, -11. / 20])), 'sgs',
        Qobj(np.diag([9. / 20, 7. / 20, 1. / 5, 0, 0]))
    )
    

def test_cosm():
    """
    Test Qobj: cosm
    """
    A = rand_herm(5)

    B = A.cosm().full()

    C = la.cosm(A.full())

    assert_(np.all((B-C)< 1e-14))


def test_sinm():
    """
    Test Qobj: sinm
    """
    A = rand_herm(5)

    B = A.sinm().full()

    C = la.sinm(A.full())

    assert_(np.all((B-C)< 1e-14))



def test_dual_channel():
    """
    Qobj: dual_chan() preserves inner products with arbitrary density ops.
    """

    def case(S, n_trials=50):        
        S = to_super(S)
        left_dims, right_dims = S.dims
        
        # Assume for the purposes of the test that S maps square operators to square operators.
        in_dim = np.prod(right_dims[0])
        out_dim = np.prod(left_dims[0])
        
        S_dual = to_super(S.dual_chan())
        
        primals = []
        duals = []
    
        for idx_trial in range(n_trials):
            X = rand_dm_ginibre(out_dim)
            X.dims = left_dims
            X = operator_to_vector(X)
            Y = rand_dm_ginibre(in_dim)
            Y.dims = right_dims
            Y = operator_to_vector(Y)

            primals.append((X.dag() * S * Y)[0, 0])
            duals.append((X.dag() * S_dual.dag() * Y)[0, 0])
    
        np.testing.assert_array_almost_equal(primals, duals)

    for subdims in [
        [2],
        [2, 2],
        [2, 3],
        [3, 5, 2]
    ]:
        dim = np.prod(subdims)
        S = rand_super_bcsz(dim)
        S.dims = [[subdims, subdims], [subdims, subdims]]
        yield case, S


def test_call():
    # Make test objects.
    psi = rand_ket(3)
    rho = rand_dm_ginibre(3)
    U = rand_unitary(3)
    S = rand_super_bcsz(3)

    # Case 0: oper(ket).
    assert U(psi) == U * psi

    # Case 1: oper(oper). Should raise TypeError.
    with expect_exception(TypeError):
        U(rho)

    # Case 2: super(ket).
    assert S(psi) == vector_to_operator(S * operator_to_vector(ket2dm(psi)))

    # Case 3: super(oper).
    assert S(rho) == vector_to_operator(S * operator_to_vector(rho))

    # Case 4: super(super). Should raise TypeError.
    with expect_exception(TypeError):
        S(S)


if __name__ == "__main__":
    run_module_suite()

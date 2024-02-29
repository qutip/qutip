import numpy as np
import scipy.linalg

import qutip
from qutip.core import data as _data
import pytest


def f(t, args):
    return t * (1 - 0.5j)


def liouvillian_ref(H, c_ops=()):
    L = -1.0j * (qutip.spre(H) - qutip.spost(H)) if H else 0
    for c in c_ops:
        if c.issuper:
            L += c
        else:
            cdc = c.dag() * c
            L += qutip.sprepost(c, c.dag())
            L -= 0.5 * (qutip.spre(cdc) + qutip.spost(cdc))
    return L


class TestMatVec:
    """
    A test class for the QuTiP function for matrix/vector conversion.
    """
    def testOperatorVector(self):
        """
        Superoperator: Operator - vector - operator conversion.
        """
        N = 3
        rho1 = qutip.rand_dm(N)
        rho2 = qutip.vector_to_operator(qutip.operator_to_vector(rho1))
        np.testing.assert_allclose(rho1.full(), rho2.full(), 1e-8)

    def testsuperrep(self):
        N = 3
        rho1 = qutip.rand_dm(N)
        as_vec = qutip.operator_to_vector(rho1)
        assert as_vec.superrep == 'super'

        with pytest.raises(TypeError) as err:
            bad_vec = as_vec.copy()
            bad_vec.superrep = "bad"
            qutip.vector_to_operator(bad_vec)
        assert err.value.args[0] == ("only defined for operator-kets "
                                     "in super format")

        with pytest.raises(TypeError) as err:
            qutip.operator_to_vector(as_vec)
        assert err.value.args[0] == ("Cannot convert object already "
                                     "in super representation")

    def testOperatorVectorTensor(self):
        """
        Superoperator: Operator - vector - operator conversion with a tensor
        product state.
        """
        Na = 3
        Nb = 2
        rhoa = qutip.rand_dm(Na)
        rhob = qutip.rand_dm(Nb)
        rho1 = qutip.tensor(rhoa, rhob)
        rho2 = qutip.vector_to_operator(qutip.operator_to_vector(rho1))
        np.testing.assert_allclose(rho1.full(), rho2.full(), 1e-8)

    def testOperatorVectorNotSquare(self):
        """
        Superoperator: Operator - vector - operator conversion for non-square
        matrix.
        """
        op1 = qutip.Qobj(np.random.rand(6).reshape((3, 2)))
        op2 = qutip.vector_to_operator(qutip.operator_to_vector(op1))
        np.testing.assert_allclose(op1.full(), op2.full(), 1e-8)

    def testOperatorSpreAppl(self):
        """
        Superoperator: apply operator and superoperator from left (spre)
        """
        N = 3
        rho = qutip.rand_dm(N)
        U = qutip.rand_unitary(N)
        rho1 = U * rho
        rho2_vec = qutip.spre(U) * qutip.operator_to_vector(rho)
        rho2 = qutip.vector_to_operator(rho2_vec)
        np.testing.assert_allclose(rho1.full(), rho2.full(), 1e-8)

    def testOperatorSpostAppl(self):
        """
        Superoperator: apply operator and superoperator from right (spost)
        """
        N = 3
        rho = qutip.rand_dm(N)
        U = qutip.rand_unitary(N)
        rho1 = rho * U
        rho2_vec = qutip.spost(U) * qutip.operator_to_vector(rho)
        rho2 = qutip.vector_to_operator(rho2_vec)
        np.testing.assert_allclose(rho1.full(), rho2.full(), 1e-8)

    def testOperatorUnitaryTransform(self):
        """
        Superoperator: Unitary transformation with operators and superoperators
        """
        N = 3
        rho = qutip.rand_dm(N)
        U = qutip.rand_unitary(N)
        rho1 = U * rho * U.dag()
        rho2_vec = qutip.sprepost(U, U.dag()) * qutip.operator_to_vector(rho)
        rho2 = qutip.vector_to_operator(rho2_vec)
        np.testing.assert_allclose(rho1.full(), rho2.full(), 1e-8)

    def testMatrixVecMat(self):
        """
        Superoperator: Conversion matrix to vector to matrix
        """
        M = _data.create(np.random.rand(10, 10))
        V = qutip.stack_columns(M)
        M2 = qutip.unstack_columns(V)
        np.testing.assert_allclose(M.to_array(), M2.to_array(), 1e-8)

    def testVecMatVec(self):
        """
        Superoperator: Conversion vector to matrix to vector
        """
        V = _data.create(np.random.rand(100, 1))
        M = qutip.unstack_columns(V)
        V2 = qutip.stack_columns(M)
        np.testing.assert_allclose(V.to_array(), V2.to_array(), 1e-8)

    def testVecMatIndexConversion(self):
        """
        Superoperator: Conversion between matrix and vector indices
        """
        N = 10
        for i in range(N * N):
            assert i == qutip.stacked_index(N, *qutip.unstacked_index(N, i))

    def testVecMatIndexCompability(self):
        """
        Superoperator: Compatibility between matrix/vector and
        corresponding index conversions.
        """
        N = 10
        M = _data.create(np.random.rand(N, N))
        V = qutip.stack_columns(M)
        for idx in range(N * N):
            i, j = qutip.unstacked_index(N, idx)
            assert V.to_array()[idx, 0] == M.to_array()[i, j]

    def test_reshuffle(self):
        U1 = qutip.rand_unitary(2)
        U2 = qutip.rand_unitary(3)
        U3 = qutip.rand_unitary(4)
        U = qutip.tensor(U1, U2, U3)
        S = qutip.to_super(U)
        S_col = qutip.reshuffle(S)
        assert S_col.dims[0] == [[2], [2], [3], [3], [4], [4]]
        assert qutip.reshuffle(S_col) == S

    def test_sprepost(self):
        U1 = qutip.rand_unitary(3)
        U2 = qutip.rand_unitary(3)
        S1 = qutip.spre(U1) * qutip.spost(U2)
        S2 = qutip.sprepost(U1, U2)
        assert S1 == S2

    @pytest.mark.parametrize("dtype", [_data.CSR, _data.Dense])
    @pytest.mark.parametrize("operation", [qutip.spre,
                                           qutip.spost,
                                           qutip.liouvillian,
                                           qutip.lindblad_dissipator])
    def test_operation_dtype(self, dtype, operation):
        """This test checks that the dtype is properly kept with the
        `operation` function."""
        U1 = qutip.rand_unitary(3).to(dtype)
        result = operation(U1)
        assert type(result.data) == dtype

    @pytest.mark.parametrize("dtype", [_data.CSR, _data.Dense])
    def test_sprepost_dtype(self, dtype):
        """This test checks that the dtype is properly kept with `sprepost`
        function."""
        U1 = qutip.rand_unitary(3).to(dtype)
        result = qutip.sprepost(U1, U1)
        assert type(result.data) == dtype

    def testLiouvillianImplem(self):
        """
        Superoperator: Randomized comparison of standard and reference
        Liouvillian functions.
        """
        N1, N2, N3 = 3, 4, 5
        a1 = qutip.tensor(qutip.rand_dm(N1, density=0.75),
                          qutip.qeye([N2, N3]))
        a2 = qutip.tensor(qutip.qeye(N1),
                          qutip.rand_dm(N2, density=0.75),
                          qutip.qeye(N3))
        a3 = qutip.tensor(qutip.qeye([N1, N2]),
                          qutip.rand_dm(N3, density=0.75))
        H = a1.dag()*a1 + a2.dag()*a2 + a3.dag()*a3
        c_ops = [np.sqrt(0.01) * a1, np.sqrt(0.025) * a2, np.sqrt(0.05) * a3]
        L1 = qutip.liouvillian(H, c_ops)
        L2 = liouvillian_ref(H, c_ops)
        np.testing.assert_allclose(L1.full(), L2.full(), 1e-8)


class TestSuper_td:
    """
    A test class for the QuTiP superoperator functions.
    """
    N = 3
    t1 = qutip.QobjEvo([qutip.qeye(N)*(1 + 0.1j),
                        [qutip.create(N) * (1 - 0.1j), f]])
    t2 = qutip.QobjEvo([qutip.destroy(N) * (1 - 0.2j)])
    t3 = qutip.QobjEvo([[qutip.num(N) * (1 + 0.2j), f]])
    q1 = qutip.qeye(N) * (1 + 0.3j)
    q2 = qutip.destroy(N) * (1 - 0.3j)
    q3 = qutip.num(N) * (1 + 0.4j)

    def test_spre_td(self):
        "Superoperator: spre, time-dependent"
        assert qutip.spre(self.t1)(0.5) == qutip.spre(self.t1(0.5))

    def test_spost_td(self):
        "Superoperator: spre, time-dependent"
        assert qutip.spost(self.t1)(0.5) == qutip.spost(self.t1(0.5))

    def test_sprepost_td(self):
        "Superoperator: sprepost, time-dependent"
        # left QobjEvo
        assert (qutip.sprepost(self.t1, self.q2)(0.5)
                == qutip.sprepost(self.t1(0.5), self.q2))
        # left QobjEvo
        assert (qutip.sprepost(self.q2, self.t1)(0.5)
                == qutip.sprepost(self.q2, self.t1(0.5)))
        # left 2 QobjEvo, one cte
        assert (qutip.sprepost(self.t1, self.t2)(0.5)
                == qutip.sprepost(self.t1(0.5), self.t2(0.5)))

    def test_operator_vector_td(self):
        "Superoperator: operator_to_vector, time-dependent"
        assert (qutip.operator_to_vector(self.t1)(0.5)
                == qutip.operator_to_vector(self.t1(0.5)))
        vec = qutip.operator_to_vector(self.t1)
        assert (qutip.vector_to_operator(vec)(0.5)
                == qutip.vector_to_operator(vec(0.5)))

    def test_liouvillian_td(self):
        "Superoperator: liouvillian, time-dependent"
        assert (qutip.liouvillian(self.t1)(0.5)
                == qutip.liouvillian(self.t1(0.5)))
        assert (qutip.liouvillian(None, [self.t2])(0.5)
                == qutip.liouvillian(None, [self.t2(0.5)]))
        assert (qutip.liouvillian(self.t1, [self.t2, self.q1, self.t3],
                                  chi=[1, 2, 3])(0.5)
                == qutip.liouvillian(self.t1(0.5),
                                     [self.t2(0.5), self.q1, self.t3(0.5)],
                                     chi=[1, 2, 3]))

    def test_lindblad_dissipator_td(self):
        "Superoperator: lindblad_dissipator, time-dependent"
        assert (qutip.lindblad_dissipator(self.t2)(0.5)
                == qutip.lindblad_dissipator(self.t2(0.5)))
        assert (qutip.lindblad_dissipator(self.t2, self.q1)(0.5)
                == qutip.lindblad_dissipator(self.t2(0.5), self.q1))
        assert (qutip.lindblad_dissipator(self.q1, self.t2)(0.5)
                == qutip.lindblad_dissipator(self.q1, self.t2(0.5)))

import pytest
import qutip
import numpy as np
from qutip.core._brtools import matmul_var_data, _EigenBasisTransform
from qutip.core._brtensor import (_br_term_dense, _br_term_sparse,
                                  _br_term_data, _BlochRedfieldElement)


def _make_rand_data(shape):
    np.random.seed(11)
    array = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    return qutip.data.Dense(array)


transform = {
    0: lambda x: x,
    1: qutip.data.transpose,
    2: qutip.data.conj,
    3: qutip.data.adjoint
}


@pytest.mark.parametrize('datatype', qutip.data.to.dtypes)
@pytest.mark.parametrize('transleft', [0, 1, 2, 3],
                         ids=['', 'transpose', 'conj', 'dag'])
@pytest.mark.parametrize('transright', [0, 1, 2, 3],
                         ids=['', 'transpose', 'conj', 'dag'])
def test_matmul_var(datatype, transleft, transright):
    shape = (5, 5)
    np.random.seed(11)
    left = qutip.data.to(datatype, _make_rand_data(shape))
    right = qutip.data.to(datatype, _make_rand_data(shape))

    expected = qutip.data.matmul(
        transform[transleft](left),
        transform[transright](right),
        ).to_array()

    computed = matmul_var_data(left, right, transleft, transright).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14)


@pytest.mark.parametrize('sparse', [False, True], ids=['Dense', 'Sparse'])
def test_eigen_transform(sparse):
    a = qutip.destroy(5)
    f = lambda t: t
    op = qutip.QobjEvo([a*a.dag(), [a+a.dag(), f]])
    eigenT = _EigenBasisTransform(op, sparse=sparse)
    evecs_qevo = eigenT.as_Qobj()

    for t in [0, 1, 1.5]:
        eigenvals, ekets = op(t).eigenstates()
        np.testing.assert_allclose(eigenvals, eigenT.eigenvalues(t),
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            np.abs(np.hstack([eket.full() for eket in ekets])),
            np.abs(eigenT.evecs(t).to_array()),
            rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(np.abs(evecs_qevo(t).full()),
                                   np.abs(eigenT.evecs(t).to_array()),
                                   rtol=1e-14, atol=1e-14)


def test_eigen_transform_ket():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.coherent(N, 1.1)

    expected = (op @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(op_diag.data, eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


def test_eigen_transform_dm():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.coherent_dm(N, 1.1)

    expected = (op @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(op_diag.data, eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


def test_eigen_transform_oper_ket():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.operator_to_vector(qutip.coherent_dm(N, 1.1))

    expected = (qutip.spre(op) @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(qutip.spre(op_diag).data,
                          eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


def test_eigen_transform_super_ops():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.sprepost(
        qutip.coherent_dm(N, 1.1),
        qutip.thermal_dm(N, 1.1)
    )

    expected = (qutip.spre(op) @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(qutip.spre(op_diag).data,
                          eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('func',
                         [_br_term_dense, _br_term_sparse, _br_term_data],
                         ids=['dense', 'sparse', 'data'])
def test_br_term_linbblad_comp(func):
    N = 5
    a = qutip.destroy(N) + qutip.destroy(N)**2 / 2
    A_op = a + a.dag()
    H = qutip.num(N)
    diag = H.eigenenergies()
    skew =  np.einsum('i,j->ji', np.ones(N), diag) - diag * np.ones((N, N))
    spectrum = (skew > 0) * 1.
    computation = func(A_op.data, spectrum, skew, 2).to_array()
    lindblad = qutip.lindblad_dissipator(a).full()
    np.testing.assert_allclose(computation, lindblad, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('cutoff', [0, 0.1, 1, 3, np.inf])
@pytest.mark.parametrize('spectra', [
    pytest.param(lambda skew: (skew > 0) * 1., id='pos_filter'),
    pytest.param(lambda skew: np.ones_like(skew), id='no_filter'),
    pytest.param(lambda skew: np.exp(skew)/10, id='smooth_filter'),
])
def test_br_term(cutoff, spectra):
    N = 5
    a = qutip.destroy(N) @ qutip.coherent_dm(N, 0.5) * 0.5
    A_op = a + a.dag()
    H = qutip.num(N)
    diag = H.eigenenergies()
    skew =  np.einsum('i,j->ji', np.ones(N), diag) - diag * np.ones((N, N))
    spectrum = spectra(skew)
    R_dense = _br_term_dense(A_op.data, spectrum, skew, cutoff).to_array()
    R_sparse = _br_term_sparse(A_op.data, spectrum, skew, cutoff).to_array()
    R_data = _br_term_data(A_op.data, spectrum, skew, cutoff).to_array()
    np.testing.assert_allclose(R_dense, R_sparse, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(R_dense, R_data, rtol=1e-14, atol=1e-14)

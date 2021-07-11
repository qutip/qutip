import numbers
import numpy as np
import pytest
import qutip


N = 5


def test_jmat_12():
    spinhalf = qutip.jmat(1 / 2.)

    paulix = np.array([[0. + 0.j, 1. + 0.j], [1. + 0.j, 0. + 0.j]])
    pauliy = np.array([[0. + 0.j, 0. - 1.j], [0. + 1.j, 0. + 0.j]])
    pauliz = np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, -1. + 0.j]])
    sigmap = np.array([[0. + 0.j, 1. + 0.j], [0. + 0.j, 0. + 0.j]])
    sigmam = np.array([[0. + 0.j, 0. + 0.j], [1. + 0.j, 0. + 0.j]])

    np.testing.assert_allclose(spinhalf[0].full() * 2, paulix)
    np.testing.assert_allclose(spinhalf[1].full() * 2, pauliy)
    np.testing.assert_allclose(spinhalf[2].full() * 2, pauliz)
    np.testing.assert_allclose(qutip.jmat(1 / 2., '+').full(), sigmap)
    np.testing.assert_allclose(qutip.jmat(1 / 2., '-').full(), sigmam)

    np.testing.assert_allclose(qutip.spin_Jx(1 / 2.).full() * 2, paulix)
    np.testing.assert_allclose(qutip.spin_Jy(1 / 2.).full() * 2, pauliy)
    np.testing.assert_allclose(qutip.spin_Jz(1 / 2.).full() * 2, pauliz)
    np.testing.assert_allclose(qutip.spin_Jp(1 / 2.).full(), sigmap)
    np.testing.assert_allclose(qutip.spin_Jm(1 / 2.).full(), sigmam)

    np.testing.assert_allclose(qutip.sigmax().full(), paulix)
    np.testing.assert_allclose(qutip.sigmay().full(), pauliy)
    np.testing.assert_allclose(qutip.sigmaz().full(), pauliz)
    np.testing.assert_allclose(qutip.sigmap().full(), sigmap)
    np.testing.assert_allclose(qutip.sigmam().full(), sigmam)

    spin_set = qutip.spin_J_set(0.5)
    for i in range(3):
        assert spinhalf[i] == spin_set[i]



def test_jmat_32():
    spin32 = qutip.jmat(3 / 2.)

    paulix32 = np.array(
        [[0.0000000 + 0.j, 0.8660254 + 0.j, 0.0000000 + 0.j, 0.0000000 + 0.j],
         [0.8660254 + 0.j, 0.0000000 + 0.j, 1.0000000 + 0.j, 0.0000000 + 0.j],
         [0.0000000 + 0.j, 1.0000000 + 0.j, 0.0000000 + 0.j, 0.8660254 + 0.j],
         [0.0000000 + 0.j, 0.0000000 + 0.j, 0.8660254 + 0.j, 0.0000000 + 0.j]])

    pauliy32 = np.array(
        [[0. + 0.j, 0. - 0.8660254j, 0. + 0.j, 0. + 0.j],
         [0. + 0.8660254j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
         [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. - 0.8660254j],
         [0. + 0.j, 0. + 0.j, 0. + 0.8660254j, 0. + 0.j]])

    pauliz32 = np.array([[1.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.0 + 0.j, -0.5 + 0.j, 0.0 + 0.j],
                         [0.0 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -1.5 + 0.j]])

    np.testing.assert_allclose(spin32[0].full(), paulix32)
    np.testing.assert_allclose(spin32[1].full(), pauliy32)
    np.testing.assert_allclose(spin32[2].full(), pauliz32)


@pytest.mark.parametrize(['spin', 'N'], [
    pytest.param(3/2., 4, id="1.5"),
    pytest.param(5/2., 6, id="2.5"),
    pytest.param(3.0, 7, id="3.0"),
])
def test_jmat_dims(spin, N):
    spin_mat = qutip.jmat(spin, '+')
    assert spin_mat.dims == [[N], [N]]
    assert spin_mat.shape == (N, N)


def test_jmat_raise():
    with pytest.raises(TypeError) as e:
        qutip.jmat(0.25)
    assert str(e.value) == 'j must be a non-negative integer or half-integer'

    with pytest.raises(TypeError) as e:
        qutip.jmat(0.5, 'zx+')
    assert str(e.value) == 'Invalid type'


@pytest.mark.parametrize(['oper_func', 'diag', 'offset', 'args'], [
    pytest.param(qutip.qeye, np.ones(N), 0, (), id="qeye"),
    pytest.param(qutip.qzero, np.zeros(N), 0, (), id="zeros"),
    pytest.param(qutip.destroy, np.arange(1, N)**0.5, 1, (), id="destroy"),
    pytest.param(qutip.destroy, np.arange(6, N+5)**0.5, 1, (5,),
                 id="destroy_offset"),
    pytest.param(qutip.create, np.arange(1, N)**0.5, -1, (), id="create"),
    pytest.param(qutip.create, np.arange(6, N+5)**0.5, -1, (5,),
                 id="create_offset"),
    pytest.param(qutip.num, np.arange(N), 0, (), id="num"),
    pytest.param(qutip.num, np.arange(5, N+5), 0, (5,), id="num_offset"),
    pytest.param(qutip.charge, np.arange(-N, N+1), 0, (), id="charge"),
    pytest.param(qutip.charge, np.arange(2, N+1)/3, 0, (2, 1/3),
                 id="charge_args"),
])
def test_diagonal_operators(oper_func, diag, offset, args):
    oper = oper_func(N, *args)
    assert oper == qutip.Qobj(np.diag(diag, offset))


@pytest.mark.parametrize(['function', 'message'], [
    (qutip.qeye, "All dimensions must be integers >= 0"),
    (qutip.destroy, "Hilbert space dimension must be integer value"),
    (qutip.create, "Hilbert space dimension must be integer value"),
], ids=["qeye", "destroy", "create"])
def test_diagonal_raise(function, message):
    with pytest.raises(ValueError) as e:
        function(2.5)
    assert str(e.value) == message


@pytest.mark.parametrize("to_test", [qutip.qzero, qutip.qeye, qutip.identity])
@pytest.mark.parametrize("dimensions", [
        2,
        [2],
        [2, 3, 4],
        1,
        [1],
        [1, 1],
    ])
def test_implicit_tensor_creation(to_test, dimensions):
    implicit = to_test(dimensions)
    if isinstance(dimensions, numbers.Integral):
        dimensions = [dimensions]
    assert implicit.dims == [dimensions, dimensions]


@pytest.mark.parametrize("to_test", [qutip.qzero, qutip.qeye, qutip.identity])
def test_super_operator_creation(to_test):
    size = 2
    implicit = to_test([[size], [size]])
    explicit = qutip.to_super(to_test(size))
    assert implicit == explicit


def test_position():
    operator = qutip.position(N)
    expected = (np.diag((np.arange(1, N) / 2)**0.5, k=-1) +
                np.diag((np.arange(1, N) / 2)**0.5, k=1))
    np.testing.assert_allclose(operator.full(), expected)


def test_momentum():
    operator = qutip.momentum(N)
    expected = (np.diag((np.arange(1, N) / 2)**0.5, k=-1) -
                np.diag((np.arange(1, N) / 2)**0.5, k=1)) * 1j
    np.testing.assert_allclose(operator.full(), expected)


def test_squeeze():
    sq = qutip.squeeze(4, 0.1 + 0.1j)
    sqmatrix = np.array([[0.99500417 + 0.j, 0.00000000 + 0.j,
                          0.07059289 - 0.07059289j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, 0.98503746 + 0.j,
                          0.00000000 + 0.j, 0.12186303 - 0.12186303j],
                         [-0.07059289 - 0.07059289j, 0.00000000 + 0.j,
                          0.99500417 + 0.j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, -0.12186303 - 0.12186303j,
                          0.00000000 + 0.j, 0.98503746 + 0.j]])
    np.testing.assert_allclose(sq.full(), sqmatrix, atol=1e-8)


def test_squeezing():
    squeeze = qutip.squeeze(4, 0.1 + 0.1j)
    a = qutip.destroy(4)
    squeezing = qutip.squeezing(a, a, 0.1 + 0.1j)
    assert squeeze == squeezing


def test_displace():
    dp = qutip.displace(4, 0.25)
    dpmatrix = np.array(
        [[0.96923323 + 0.j, -0.24230859 + 0.j, 0.04282883 + 0.j, -
          0.00626025 + 0.j],
         [0.24230859 + 0.j, 0.90866411 + 0.j, -0.33183303 +
          0.j, 0.07418172 + 0.j],
         [0.04282883 + 0.j, 0.33183303 + 0.j, 0.84809499 +
          0.j, -0.41083747 + 0.j],
         [0.00626025 + 0.j, 0.07418172 + 0.j, 0.41083747 + 0.j,
          0.90866411 + 0.j]])
    np.testing.assert_allclose(dp.full(), dpmatrix, atol=1e-8)


@pytest.mark.parametrize("shift", [1, 2])
def test_tunneling(shift):
    N = 10
    tn = qutip.tunneling(N, shift)
    tn_matrix = np.diag(np.ones(N - shift), k=shift)
    tn_matrix += tn_matrix.T
    np.testing.assert_allclose(tn.full(), tn_matrix)


def test_commutator():
    A = qutip.qeye(N)
    B = qutip.destroy(N)
    assert qutip.commutator(A, B) == qutip.qzero(N)

    sx = qutip.sigmax()
    sy = qutip.sigmay()
    assert qutip.commutator(sx, sy) / 2 == (qutip.sigmaz() * 1j)

    A = qutip.qeye(N)
    B = qutip.destroy(N)
    assert qutip.commutator(A, B, 'anti') == qutip.destroy(N) * 2

    sx = qutip.sigmax()
    sy = qutip.sigmay()
    assert qutip.commutator(sx, sy, 'anti') == qutip.qzero(2)

    with pytest.raises(TypeError) as e:
        qutip.commutator(sx, sy, 'something')
    assert str(e.value).startswith("Unknown commutator kind")

def test_qutrit_ops():
    ops = qutip.qutrit_ops()
    assert qutip.qeye(3) == sum(ops[:3])
    np.testing.assert_allclose(np.diag([1, 1], k=1), sum(ops[3:5]).full())
    expected = np.zeros((3, 3))
    expected[2, 0] = 1
    np.testing.assert_allclose(expected, ops[5].full())

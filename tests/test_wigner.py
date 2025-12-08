import pytest
import numpy as np
from scipy.integrate import trapezoid
import itertools
from scipy.special import laguerre
from numpy.random import rand
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose

import qutip
from qutip.core.states import coherent, fock, ket, bell_state
from qutip.wigner import wigner, wigner_transform, _parity
from qutip.random_objects import rand_dm, rand_ket


class TestHusimiQ:
    @pytest.mark.parametrize('xs', ["", 1, None], ids=['str', 'int', 'none'])
    def test_failure_if_non_arraylike_coordinates(self, xs):
        state = qutip.rand_ket(4)
        valid = np.linspace(-1, 1, 5)
        with pytest.raises(TypeError) as e:
            qutip.qfunc(state, xs, valid)
        assert "must be array-like" in e.value.args[0]
        with pytest.raises(TypeError) as e:
            qutip.qfunc(state, valid, xs)
        assert "must be array-like" in e.value.args[0]
        with pytest.raises(TypeError) as e:
            qutip.QFunc(xs, valid)
        assert "must be array-like" in e.value.args[0]
        with pytest.raises(TypeError) as e:
            qutip.QFunc(valid, xs)
        assert "must be array-like" in e.value.args[0]

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_failure_if_coordinates_not_1d(self, ndim):
        state = qutip.rand_ket(4)
        valid = np.linspace(-1, 1, 5)
        bad = valid.reshape((-1,) + (1,)*(ndim - 1))
        with pytest.raises(ValueError) as e:
            qutip.qfunc(state, bad, valid)
        assert "must be 1D" in e.value.args[0]
        with pytest.raises(ValueError) as e:
            qutip.qfunc(state, valid, bad)
        assert "must be 1D" in e.value.args[0]
        with pytest.raises(ValueError) as e:
            qutip.QFunc(bad, valid)
        assert "must be 1D" in e.value.args[0]
        with pytest.raises(ValueError) as e:
            qutip.QFunc(valid, bad)
        assert "must be 1D" in e.value.args[0]

    @pytest.mark.parametrize('dm', [True, False], ids=['dm', 'ket'])
    def test_failure_if_tensor_hilbert_space(self, dm):
        if dm:
            state = qutip.rand_dm([2, 2])
        else:
            state = qutip.rand_ket([2, 2])
        xs = np.linspace(-1, 1, 5)
        with pytest.raises(ValueError) as e:
            qutip.qfunc(state, xs, xs)
        assert "must not have tensor structure" in e.value.args[0]
        with pytest.raises(ValueError) as e:
            qutip.QFunc(xs, xs)(state)
        assert "must not have tensor structure" in e.value.args[0]

    def test_QFunc_raises_if_insufficient_memory(self):
        xs = np.linspace(-1, 1, 11)
        state = qutip.rand_ket(4)
        qfunc = qutip.QFunc(xs, xs, memory=0)
        with pytest.raises(MemoryError) as e:
            qfunc(state)
        assert e.value.args[0].startswith("Refusing to precompute")

    def test_qfunc_warns_if_insufficient_memory(self):
        xs = np.linspace(-1, 1, 11)
        state = qutip.rand_dm(4)
        with pytest.warns(UserWarning) as e:
            qutip.qfunc(state, xs, xs, precompute_memory=0)
        assert (
            e[0].message.args[0]
            .startswith("Falling back to iterative algorithm")
        )

    @pytest.mark.parametrize('obj', [
        pytest.param(np.eye(2, dtype=np.complex128), id='ndarray'),
        pytest.param([[1, 0], [0, 1]], id='list'),
        pytest.param(1, id='int'),
    ])
    def test_failure_if_not_a_Qobj(self, obj):
        xs = np.linspace(-1, 1, 11)
        with pytest.raises(TypeError) as e:
            qutip.qfunc(obj, xs, xs)
        assert e.value.args[0].startswith("state must be Qobj")
        qfunc = qutip.QFunc(xs, xs)
        with pytest.raises(TypeError) as e:
            qfunc(obj)
        assert e.value.args[0].startswith("state must be Qobj")

    # Use indirection so that the tests can still be collected if there's a bug
    # in the generating QuTiP functions.
    @pytest.mark.parametrize('state', [
        pytest.param(lambda: qutip.rand_super(2), id='super'),
        pytest.param(lambda: qutip.rand_ket(2).dag(), id='bra'),
        pytest.param(lambda: 1j*qutip.rand_dm(2), id='non-dm operator'),
        pytest.param(lambda: qutip.Qobj([[1, 0], [0, 0]], dims=[[2], [2, 1]]),
                     id='nonsquare dm'),
        pytest.param(lambda: qutip.operator_to_vector(qutip.qeye(2)),
                     id='operator-ket'),
        pytest.param(lambda: qutip.operator_to_vector(qutip.qeye(2)).dag(),
                     id='operator-bra'),
    ])
    def test_failure_if_not_a_state(self, state):
        xs = np.linspace(-1, 1, 11)
        state = state()
        with pytest.raises(ValueError) as e:
            qutip.qfunc(state, xs, xs)
        assert (
            e.value.args[0].startswith("state must be a ket or density matrix")
        )
        qfunc = qutip.QFunc(xs, xs)
        with pytest.raises(ValueError) as e:
            qfunc(state)
        assert (
            e.value.args[0].startswith("state must be a ket or density matrix")
        )

    @pytest.mark.parametrize('g', [
        pytest.param(np.sqrt(2), id='natural units'),
        pytest.param(1, id='arb units'),
    ])
    @pytest.mark.parametrize('n_ys', [5, 101])
    @pytest.mark.parametrize('n_xs', [5, 101])
    @pytest.mark.parametrize('dm', [True, False], ids=['dm', 'ket'])
    @pytest.mark.parametrize('size', [5, 32])
    def test_function_and_class_are_equivalent(self, size, dm, n_xs, n_ys, g):
        xs = np.linspace(-1, 1, n_xs)
        ys = np.linspace(0, 2, n_ys)
        state = qutip.rand_dm(size) if dm else qutip.rand_ket(size)
        function = qutip.qfunc(state, xs, ys, g)
        class_ = qutip.QFunc(xs, ys, g)(state)
        np.testing.assert_allclose(function, class_)

    @pytest.mark.parametrize('g', [
        pytest.param(np.sqrt(2), id='natural units'),
        pytest.param(1, id='arb units'),
    ])
    @pytest.mark.parametrize('n_ys', [5, 101])
    @pytest.mark.parametrize('n_xs', [5, 101])
    @pytest.mark.parametrize('size', [5, 32])
    def test_iterate_and_precompute_are_equivalent(self, size, n_xs, n_ys, g):
        xs = np.linspace(-1, 1, n_xs)
        ys = np.linspace(0, 2, n_ys)
        state = qutip.rand_dm(size)
        iterate = qutip.qfunc(state, xs, ys, g, precompute_memory=None)
        precompute = qutip.qfunc(state, xs, ys, g, precompute_memory=np.inf)
        np.testing.assert_allclose(iterate, precompute)

    @pytest.mark.parametrize('initial_size', [5, 8])
    @pytest.mark.parametrize('dm', [True, False], ids=['dm', 'ket'])
    def test_same_class_can_take_many_sizes(self, dm, initial_size):
        xs = np.linspace(-1, 1, 11)
        ys = np.linspace(0, 2, 11)
        shape = np.meshgrid(xs, ys)[0].shape
        sizes = initial_size + np.array([0, 1, -1, 4])
        qfunc = qutip.QFunc(xs, ys)
        for size in sizes:
            state = qutip.rand_dm(size) if dm else qutip.rand_ket(size)
            out = qfunc(state)
            assert isinstance(out, np.ndarray)
            assert out.shape == shape

    @pytest.mark.parametrize('dm_first', [True, False])
    def test_same_class_can_mix_ket_and_dm(self, dm_first):
        dms = [True, False, True, False]
        if not dm_first:
            dms = dms[::-1]
        xs = np.linspace(-1, 1, 11)
        ys = np.linspace(0, 2, 11)
        shape = np.meshgrid(xs, ys)[0].shape
        qfunc = qutip.QFunc(xs, ys)
        for dm in dms:
            state = qutip.rand_dm(4) if dm else qutip.rand_ket(4)
            out = qfunc(state)
            assert isinstance(out, np.ndarray)
            assert out.shape == shape

    @pytest.mark.parametrize('n_ys', [5, 101])
    @pytest.mark.parametrize('n_xs', [5, 101])
    @pytest.mark.parametrize('mix', [0.1, 0.5])
    def test_qfunc_is_linear(self, n_xs, n_ys, mix):
        xs = np.linspace(-1, 1, n_xs)
        ys = np.linspace(-1, 1, n_ys)
        qfunc = qutip.QFunc(xs, ys)
        left, right = qutip.rand_dm(5), qutip.rand_dm(5)
        qleft, qright = qfunc(left), qfunc(right)
        qboth = qfunc(mix*left + (1-mix)*right)
        np.testing.assert_allclose(mix*qleft + (1-mix)*qright, qboth)

    @pytest.mark.parametrize('n_ys', [5, 101])
    @pytest.mark.parametrize('n_xs', [5, 101])
    @pytest.mark.parametrize('size', [5, 32])
    def test_ket_and_dm_give_same_result(self, n_xs, n_ys, size):
        xs = np.linspace(-1, 1, n_xs)
        ys = np.linspace(-1, 1, n_ys)
        state = qutip.rand_ket(size)
        qfunc = qutip.QFunc(xs, ys)
        np.testing.assert_allclose(qfunc(state), qfunc(state.proj()))

    @pytest.mark.parametrize('g', [
        pytest.param(np.sqrt(2), id='natural units'),
        pytest.param(1, id='arb units'),
    ])
    @pytest.mark.parametrize('ys', [
        pytest.param(np.linspace(-1, 1, 5), id='(-1,1,5)'),
        pytest.param(np.linspace(0, 2, 3), id='(0,2,3)'),
    ])
    @pytest.mark.parametrize('xs', [
        pytest.param(np.linspace(-1, 1, 5), id='(-1,1,5)'),
        pytest.param(np.linspace(0, 2, 3), id='(0,2,3)'),
    ])
    @pytest.mark.parametrize('size', [3, 5])
    def test_against_naive_implementation(self, xs, ys, g, size):
        state = qutip.rand_dm(size)
        state_np = state.full()
        x, y = np.meshgrid(xs, ys)
        alphas = 0.5*g * (x + 1j*y)
        naive = np.empty(alphas.shape, dtype=np.float64)
        for i, alpha in enumerate(alphas.flat):
            coh = qutip.coherent(size, alpha, method='analytic').full()
            naive.flat[i] = (coh.conj().T @ state_np @ coh).real[0, 0]
        naive *= (0.5*g)**2 / np.pi
        np.testing.assert_allclose(naive, qutip.qfunc(state, xs, ys, g))
        np.testing.assert_allclose(naive, qutip.QFunc(xs, ys, g)(state))


def test_wigner_bell1_su2parity():
    """wigner: testing the SU2 parity of the first Bell state.
    """
    psi = bell_state('00')

    steps = 25
    theta = np.tile(np.linspace(0, np.pi, steps), 2).reshape(2, steps)
    phi = np.tile(np.linspace(0, 2 * np.pi, steps), 2).reshape(2, steps)
    slicearray = ['l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = np.real(((1 + np.sqrt(3)
                                            * np.cos(theta[0, t]))
                                           * (1 + np.sqrt(3)
                                           * np.cos(theta[1, t]))
                                           + 3 * (np.sin(theta[0, t])
                                           * np.exp(-1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(-1j * phi[1, p])
                                           + np.sin(theta[0, t])
                                           * np.exp(1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(1j * phi[1, p]))
                                           + (1 - np.sqrt(3)
                                           * np.cos(theta[0, t]))
                                           * (1 - np.sqrt(3)
                                           * np.cos(theta[1, t]))) / 8.)

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert (np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


@pytest.mark.slow
def test_wigner_bell4_su2parity():
    """wigner: testing the SU2 parity of the fourth Bell state.
    """
    psi = bell_state('11')

    steps = 25
    slicearray = ['l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = -0.5

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert (np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


@pytest.mark.slow
def test_wigner_bell4_fullparity():
    """wigner: testing the parity of the fourth Bell state using the parity of
    the full space.
    """
    psi = bell_state('11')

    steps = 25
    slicearray = ['l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = -0.30901699

    wigner_theo = wigner_transform(psi, 0.5, True, steps, slicearray)
    assert (np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-4)


def test_parity():
    """wigner: testing the parity function.
    """
    j = 0.5
    assert (_parity(2, j)[0, 0] - (1 - np.sqrt(3)) / 2. < 1e-11)
    assert (_parity(2, j)[0, 1] < 1e-11)
    assert (_parity(2, j)[1, 1] - (1 + np.sqrt(3)) / 2. < 1e-11)
    assert (_parity(2, j)[1, 0] < 1e-11)


@pytest.mark.slow
def test_wigner_pure_su2():
    """wigner: testing the SU2 wigner transformation of a pure state.
    """
    psi = (ket([1]))
    steps = 25
    theta = np.linspace(0, np.pi, steps)
    phi = np.linspace(0, 2 * np.pi, steps)
    theta = theta[None, :]
    phi = phi[None, :]
    slicearray = ['l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = (1 + np.sqrt(3) * np.cos(theta[0, t])) / 2.

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert (np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


@pytest.mark.slow
def test_wigner_ghz_su2parity():
    """wigner: testing the SU2 wigner transformation of the GHZ state.
    """
    psi = (ket([0, 0, 0]) + ket([1, 1, 1])) / np.sqrt(2)

    steps = 25
    N = 3
    theta = np.tile(np.linspace(0, np.pi, steps), N).reshape(N, steps)
    phi = np.tile(np.linspace(0, 2 * np.pi, steps), N).reshape(N, steps)
    slicearray = ['l', 'l', 'l']

    wigner_analyt = np.zeros((steps, steps))
    for t in range(steps):
        for p in range(steps):
            wigner_analyt[t, p] = np.real(((1 + np.sqrt(3)*np.cos(theta[0, t]))
                                           * (1 + np.sqrt(3)
                                           * np.cos(theta[1, t]))
                                           * (1 + np.sqrt(3)
                                           * np.cos(theta[2, t]))
                                           + 3**(3 / 2) * (np.sin(theta[0, t])
                                           * np.exp(-1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(-1j * phi[1, p])
                                           * np.sin(theta[2, t])
                                           * np.exp(-1j * phi[2, p])
                                           + np.sin(theta[0, t])
                                           * np.exp(1j * phi[0, p])
                                           * np.sin(theta[1, t])
                                           * np.exp(1j * phi[1, p])
                                           * np.sin(theta[2, t])
                                           * np.exp(1j * phi[2, p]))
                                           + (1 - np.sqrt(3)
                                           * np.cos(theta[0, t]))
                                           * (1 - np.sqrt(3)
                                           * np.cos(theta[1, t]))
                                           * (1 - np.sqrt(3)
                                           * np.cos(theta[2, t]))) / 16.)

    wigner_theo = wigner_transform(psi, 0.5, False, steps, slicearray)

    assert (np.sum(np.abs(wigner_analyt - wigner_theo)) < 1e-11)


@pytest.mark.slow
def test_angle_slicing():
    """wigner: tests angle slicing.
    """
    psi1 = bell_state('00')
    psi2 = bell_state('01')
    psi3 = bell_state('10')
    psi4 = bell_state('11')

    steps = 25
    j = 0.5

    wigner1 = wigner_transform(psi1, j, False, steps, ['l', 'l'])
    wigner2 = wigner_transform(psi2, j, False, steps, ['l', 'z'])
    wigner3 = wigner_transform(psi3, j, False, steps, ['l', 'x'])
    wigner4 = wigner_transform(psi4, j, False, steps, ['l', 'y'])

    assert (np.sum(np.abs(wigner2 - wigner1)) < 1e-11)
    assert (np.sum(np.abs(wigner3 - wigner2)) < 1e-11)
    assert (np.sum(np.abs(wigner4 - wigner3)) < 1e-11)
    assert (np.sum(np.abs(wigner4 - wigner1)) < 1e-11)


def test_wigner_coherent():
    "wigner: test wigner function calculation for coherent states"
    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 20
    beta = rand() + rand() * 1.0j
    psi = coherent(N, beta)

    # calculate the wigner function using qutip and analytic formula
    W_qutip = wigner(psi, xvec, yvec, g=2)
    W_qutip_cl = wigner(psi, xvec, yvec, g=2, method='clenshaw')
    W_analytic = 2 / np.pi * np.exp(-2 * abs(a - beta) ** 2)

    # check difference
    assert (np.sum(abs(W_qutip - W_analytic) ** 2) < 1e-4)
    assert (np.sum(abs(W_qutip_cl - W_analytic) ** 2) < 1e-4)

    # check normalization
    assert (np.sum(W_qutip) * dx * dy - 1.0 < 1e-8)
    assert (np.sum(W_qutip_cl) * dx * dy - 1.0 < 1e-8)
    assert (np.sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_fock():
    "wigner: test wigner function calculation for Fock states"

    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in [2, 3, 4, 5, 6]:

        psi = fock(N, n)

        # calculate the wigner function using qutip and analytic formula
        W_qutip = wigner(psi, xvec, yvec, g=2)
        W_qutip_cl = wigner(psi, xvec, yvec, g=2, method='clenshaw')
        W_qutip_sparse = wigner(psi, xvec, yvec, g=2, sparse=True, method='clenshaw')
        W_analytic = 2 / np.pi * (-1) ** n * \
            np.exp(-2 * abs(a) ** 2) * np.polyval(laguerre(n), 4 * abs(a) ** 2)

        # check difference
        assert (np.sum(abs(W_qutip - W_analytic)) < 1e-4)
        assert (np.sum(abs(W_qutip_cl - W_analytic)) < 1e-4)
        assert (np.sum(abs(W_qutip_sparse - W_analytic)) < 1e-4)

        # check normalization
        assert (np.sum(W_qutip) * dx * dy - 1.0 < 1e-8)
        assert (np.sum(W_qutip_cl) * dx * dy - 1.0 < 1e-8)
        assert (np.sum(W_qutip_sparse) * dx * dy - 1.0 < 1e-8)
        assert (np.sum(W_analytic) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_dm():
    "wigner: compare wigner methods for random density matrices"

    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    # a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        rho = rand_dm(N, density=0.5 + rand() / 2)

        # calculate the wigner function using qutip and analytic formula
        W_qutip1 = wigner(rho, xvec, yvec, g=2)
        W_qutip2 = wigner(rho, xvec, yvec, g=2, method='laguerre')

        # check difference
        assert (np.sum(abs(W_qutip1 - W_qutip1)) < 1e-4)

        # check normalization
        assert (np.sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert (np.sum(W_qutip2) * dx * dy - 1.0 < 1e-8)


def test_wigner_compare_methods_ket():
    "wigner: compare wigner methods for random state vectors"

    xvec = np.linspace(-5.0, 5.0, 100)
    yvec = xvec

    X, Y = np.meshgrid(xvec, yvec)

    # a = X + 1j * Y  # consistent with g=2 option to wigner function

    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]

    N = 15

    for n in range(10):
        # try ten different random density matrices

        psi = rand_ket(N, density=0.5 + rand() / 2)

        # calculate the wigner function using qutip and analytic formula
        W_qutip1 = wigner(psi, xvec, yvec, g=2)
        W_qutip2 = wigner(psi, xvec, yvec, g=2, sparse=True)

        # check difference
        assert (np.sum(abs(W_qutip1 - W_qutip2)) < 1e-4)

        # check normalization
        assert (np.sum(W_qutip1) * dx * dy - 1.0 < 1e-8)
        assert (np.sum(W_qutip2) * dx * dy - 1.0 < 1e-8)


def test_wigner_fft_comparse_ket():
    "Wigner: Compare Wigner fft and iterative for rand. ket"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_ket(N)

        Wfft, yvec = wigner(rho, xvec, xvec, method='fft')
        W = wigner(rho, xvec, yvec, method='iterative')

        Wdiff = abs(W - Wfft)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_fft_comparse_dm():
    "Wigner: Compare Wigner fft and iterative for rand. dm"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wfft, yvec = wigner(rho, xvec, xvec, method='fft')
        W = wigner(rho, xvec, yvec, method='iterative')

        Wdiff = abs(W - Wfft)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_clenshaw_iter_dm():
    "Wigner: Compare Wigner clenshaw and iterative for rand. dm"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wclen = wigner(rho, xvec, xvec, method='clenshaw')
        W = wigner(rho, xvec, xvec, method='iterative')

        Wdiff = abs(W - Wclen)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)


def test_wigner_clenshaw_sp_iter_dm():
    "Wigner: Compare Wigner sparse clenshaw and iterative for rand. dm"
    N = 20
    xvec = np.linspace(-10, 10, 128)
    for i in range(3):
        rho = rand_dm(N)

        Wclen = wigner(rho, xvec, xvec, method='clenshaw', sparse=True)
        W = wigner(rho, xvec, xvec, method='iterative')

        Wdiff = abs(W - Wclen)
        assert_equal(np.sum(abs(Wdiff)) < 1e-7, True)

@pytest.mark.parametrize(['spin'], [
    pytest.param(1/2, id="spin-one-half"),
    pytest.param(3, id="spin-three"),
    pytest.param(13/2, id="spin-thirteen-half"),
    pytest.param(7, id="spin-seven")
])
@pytest.mark.parametrize("pure", [
    pytest.param("pure", id="pure"),
    pytest.param("herm", id="mixed")
])
def test_spin_q_function(spin, pure):
    d = int(2*spin + 1)
    rho = rand_dm(d, distribution=pure)

    # Points at which to evaluate the spin Q function
    theta = np.linspace(0, np.pi, 16, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, 32, endpoint=True)
    Q, _, _ = qutip.spin_q_function(rho, theta, phi)

    for k, (phi_prime, theta_prime) in enumerate(itertools.product(phi, theta)):
        state = qutip.spin_coherent(spin, theta_prime, phi_prime)
        direct_Q = abs(state.dag() * rho * state)
        assert_almost_equal(Q.flat[k], direct_Q, decimal=9)

@pytest.mark.parametrize(['spin'], [
    pytest.param(1/2, id="spin-one-half"),
    pytest.param(3, id="spin-three"),
    pytest.param(13/2, id="spin-thirteen-half"),
    pytest.param(7, id="spin-seven")
])
@pytest.mark.parametrize("pure", [
    pytest.param("pure", id="pure"),
    pytest.param("herm", id="mixed")
])
def test_spin_q_function_normalized(spin, pure):
    d = int(2 * spin + 1)
    rho = rand_dm(d, distribution=pure)

    # Points at which to evaluate the spin Q function
    theta = np.linspace(0, np.pi, 128, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    Q, THETA, _ = qutip.spin_q_function(rho, theta, phi)

    norm = d / (4 * np.pi) * trapezoid(
        trapezoid(Q * np.sin(THETA), theta), phi
    )
    assert_allclose(norm, 1, atol=2e-4)


@pytest.mark.parametrize(["spin"], [
    pytest.param(1/2, id="spin-one-half"),
    pytest.param(1, id="spin-one"),
    pytest.param(3/2, id="spin-three-half"),
    pytest.param(2, id="spin-two")
])
@pytest.mark.parametrize("pure", [
    pytest.param("pure", id="pure"),
    pytest.param("herm", id="mixed")
])
def test_spin_wigner_normalized(spin, pure):
    d = int(2*spin + 1)
    rho = rand_dm(d, distribution=pure)

    # Points at which to evaluate the spin Wigner function
    theta = np.linspace(0, np.pi, 256, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, 512, endpoint=True)
    W, THETA, PHI = qutip.spin_wigner(rho, theta, phi)

    norm = trapezoid(
        trapezoid(W * np.sin(THETA) * np.sqrt(d / (4*np.pi)), theta), phi
    )
    assert_almost_equal(norm, 1, decimal=4)

@pytest.mark.parametrize(['spin'], [
    pytest.param(1 / 2, id="spin-one-half"),
    pytest.param(1, id="spin-one"),
    pytest.param(3 / 2, id="spin-three-half"),
    pytest.param(2, id="spin-two")
])
@pytest.mark.parametrize("pure", [
    pytest.param("pure", id="pure"),
    pytest.param("herm", id="mixed")
])
def test_spin_wigner_overlap(spin, pure, n=5):
    d = int(2*spin + 1)
    rho = rand_dm(d, distribution=pure)

    # Points at which to evaluate the spin Wigner function
    theta = np.linspace(0, np.pi, 256, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, 512, endpoint=True)
    W, THETA, _ = qutip.spin_wigner(rho, theta, phi)

    for k in range(n):
        test_state = rand_dm(d)
        state_overlap = (test_state*rho).tr().real

        W_state, _, _ = qutip.spin_wigner(test_state, theta, phi)
        W_overlap = trapezoid(
            trapezoid(W_state * W * np.sin(THETA), theta), phi).real
        assert_almost_equal(W_overlap, state_overlap, decimal=4)

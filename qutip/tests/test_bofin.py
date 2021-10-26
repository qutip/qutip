"""
Tests for the Bosonic HEOM solvers.
"""

import numpy as np
import pytest
from numpy.linalg import eigvalsh
from scipy.integrate import quad

from qutip import (
    Qobj, QobjEvo, sigmaz, sigmax, basis, destroy, expect, Options
)
from qutip.nonmarkov.bofin import (
    _convert_h_sys,
    _convert_coup_op,
    BathExponent,
    Bath,
    HierarchyADOs,
    BosonicHEOMSolver,
    FermionicHEOMSolver,
    HSolverDL,
)
from qutip.states import enr_state_dictionaries


def check_exponent(
    exp, type, dim, Q, ck, vk, ck2=None, sigma_bar_k_offset=None,
):
    """ Check the attributes of a BathExponent. """
    assert exp.type is BathExponent.types[type]
    assert exp.dim == dim
    assert exp.Q == Q
    assert exp.ck == ck
    assert exp.vk == vk
    assert exp.ck2 == ck2
    assert exp.sigma_bar_k_offset == sigma_bar_k_offset


class TestBathExponent:
    def test_create(self):
        exp_r = BathExponent("R", None, Q=None, ck=1.0, vk=2.0)
        check_exponent(exp_r, "R", None, Q=None, ck=1.0, vk=2.0)

        exp_i = BathExponent("I", None, Q=None, ck=1.0, vk=2.0)
        check_exponent(exp_i, "I", None, Q=None, ck=1.0, vk=2.0)

        exp_i = BathExponent("RI", None, Q=None, ck=1.0, vk=2.0, ck2=3.0)
        check_exponent(exp_i, "RI", None, Q=None, ck=1.0, vk=2.0, ck2=3.0)

        exp_p = BathExponent(
            "+", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=-1,
        )
        check_exponent(
            exp_p, "+", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=-1,
        )

        exp_m = BathExponent(
            "-", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=+1,
        )
        check_exponent(
            exp_m, "-", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=+1,
        )

        for exp_type, kw in [
            ("R", {}),
            ("I", {}),
            ("+", {"sigma_bar_k_offset": 1}),
            ("-", {"sigma_bar_k_offset": -1}),
        ]:
            with pytest.raises(ValueError) as err:
                BathExponent(
                    exp_type, None, Q=None, ck=1.0, vk=2.0, ck2=3.0, **kw,
                )
            assert str(err.value) == (
                "Second co-efficient (ck2) should only be specified for RI"
                " bath exponents"
            )

        for exp_type, kw in [("R", {}), ("I", {}), ("RI", {"ck2": 3.0})]:
            with pytest.raises(ValueError) as err:
                BathExponent(
                    exp_type, None, Q=None, ck=1.0, vk=2.0,
                    sigma_bar_k_offset=1, **kw,
                )
            assert str(err.value) == (
                "Offset of sigma bar (sigma_bar_k_offset) should only be"
                " specified for + and - bath exponents"
            )


class TestBath:
    def test_create(self):
        exp_r = BathExponent("R", None, Q=None, ck=1.0, vk=2.0)
        exp_i = BathExponent("I", None, Q=None, ck=1.0, vk=2.0)
        bath = Bath([exp_r, exp_i])
        assert bath.exponents == [exp_r, exp_i]


class TestHierarchyADOs:
    def mk_exponents(self, dims):
        return [
            BathExponent("I", dim, Q=None, ck=1.0, vk=2.0) for dim in dims
        ]

    def test_create(self):
        exponents = self.mk_exponents([2, 3])
        ados = HierarchyADOs(exponents, cutoff=2)

        assert ados.exponents == exponents
        assert ados.cutoff == 2

        assert ados.dims == [2, 3]
        assert ados.vk == [2.0, 2.0]
        assert ados.ck == [1.0, 1.0]
        assert ados.ck2 == [None, None]
        assert ados.sigma_bar_k_offset == [None, None]

        assert ados.labels == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
        ]

    def test_state_idx(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.idx((0, 0)) == 0
        assert ados.idx((0, 1)) == 1
        assert ados.idx((0, 2)) == 2
        assert ados.idx((1, 0)) == 3
        assert ados.idx((1, 1)) == 4

    def test_next(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.next((0, 0), 0) == (1, 0)
        assert ados.next((0, 0), 1) == (0, 1)
        assert ados.next((1, 0), 0) is None
        assert ados.next((1, 0), 1) == (1, 1)
        assert ados.next((1, 1), 1) is None

    def test_prev(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.prev((0, 0), 0) is None
        assert ados.prev((0, 0), 1) is None
        assert ados.prev((1, 0), 0) == (0, 0)
        assert ados.prev((0, 1), 1) == (0, 0)
        assert ados.prev((1, 1), 1) == (1, 0)
        assert ados.prev((0, 2), 1) == (0, 1)


class TestParameterConversionUtilities:
    def test_convert_h_sys(self):
        """Tests the function for checking system Hamiltonian"""
        _convert_h_sys(sigmax())
        _convert_h_sys([sigmax(), sigmaz()])
        _convert_h_sys([[sigmax(), np.sin], [sigmaz(), np.cos]])
        _convert_h_sys([[sigmax(), np.sin], [sigmaz(), np.cos]])
        _convert_h_sys(QobjEvo([sigmaz(), sigmax(), sigmaz()]))

        with pytest.raises(TypeError) as err:
            _convert_h_sys(sigmax().full())
        assert str(err.value) == (
            "Hamiltonian (H_sys) has unsupported type: <class 'numpy.ndarray'>"
        )

        with pytest.raises(ValueError) as err:
            _convert_h_sys([[1, 0], [0, 1]])
        assert str(err.value) == (
            "Hamiltonian (H_sys) of type list cannot be converted to QObjEvo"
        )
        assert isinstance(err.value.__cause__, TypeError)
        assert str(err.value.__cause__) == "Incorrect Q_object specification"

    def test_convert_coup_op(self):
        sx = sigmax()
        sz = sigmaz()
        assert _convert_coup_op(sx, 2) == [sx, sx]
        assert _convert_coup_op([sx, sz], 2) == [sx, sz]

        with pytest.raises(TypeError) as err:
            _convert_coup_op(3, 1)
        assert str(err.value) == (
            "Coupling operator (coup_op) must be a Qobj or a list of Qobjs"
        )

        with pytest.raises(TypeError) as err:
            _convert_coup_op([sx, 1], 2)
        assert str(err.value) == (
            "Coupling operator (coup_op) must be a Qobj or a list of Qobjs"
        )

        with pytest.raises(ValueError) as err:
            _convert_coup_op([sx, sz], 3)
        assert str(err.value) == "Expected 3 coupling operators"


class TestBosonicHEOMSolver:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    @pytest.mark.parametrize(['fake_timedep'], [
        pytest.param(False, id="static"),
        pytest.param(True, id="timedep"),
    ])
    def test_run_pure_dephasing_model(self, fake_timedep):
        """ Compare with pure-dephasing analytical result. """
        tol = 1e-3
        gamma = 0.05
        lam = 0.025
        lam_c = lam / np.pi
        T = 1 / 0.95
        times = np.linspace(0, 10, 21)

        def _integrand(omega, t):
            J = 2*lam_c * omega * gamma / (omega**2 + gamma**2)
            return (-4 * J * (1 - np.cos(omega*t))
                    / (np.tanh(0.5*omega / T) * omega**2))

        # Calculate the analytical results by numerical integration
        expected = [0.5*np.exp(quad(_integrand, 0, np.inf, args=(t,))[0])
                    for t in times]

        Nk = 2
        ckAR = [lam * gamma * (1/np.tan(gamma / (2 * T)))]
        ckAR.extend([
            (4 * lam * gamma * T * 2 * np.pi * k * T /
                ((2 * np.pi * k * T)**2 - gamma**2))
            for k in range(1, Nk + 1)
        ])
        vkAR = [gamma]
        vkAR.extend([2 * np.pi * k * T for k in range(1, Nk + 1)])
        ckAI = [lam * gamma * (-1.0)]
        vkAI = [gamma]

        H_sys = Qobj(np.zeros((2, 2)))
        if fake_timedep:
            H_sys = [H_sys]
        Q = sigmaz()

        NR = len(ckAR)
        NI = len(ckAI)
        Q2 = [Q for kk in range(NR+NI)]
        initial_state = 0.5*Qobj(np.ones((2, 2)))
        projector = basis(2, 0) * basis(2, 1).dag()
        options = Options(nsteps=15000, store_states=True)

        hsolver = BosonicHEOMSolver(
            H_sys, Q2, ckAR, ckAI, vkAR, vkAI,
            14, options=options,
        )
        test = expect(hsolver.run(initial_state, times).states, projector)

        np.testing.assert_allclose(test, expected, atol=tol)


class TestHSolverDL:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    @pytest.mark.parametrize(['bnd_cut_approx', 'tol', 'fake_timedep'], [
        pytest.param(True, 1e-4, False, id="bnd_cut_approx_static"),
        pytest.param(False,  1e-3, False, id="no_bnd_cut_approx_static"),
        pytest.param(True, 1e-4, True, id="bnd_cut_approx_timedep"),
        pytest.param(False,  1e-3, True, id="no_bnd_cut_approx_timedep"),
    ])
    def test_run_pure_dephasing_model(self, bnd_cut_approx, tol, fake_timedep):
        """ Compare with pure-dephasing analytical result. """
        cut_frequency = 0.05
        coupling_strength = 0.025
        lam_c = coupling_strength / np.pi
        temperature = 1 / 0.95
        times = np.linspace(0, 10, 21)

        def _integrand(omega, t):
            J = 2*lam_c * omega * cut_frequency / (omega**2 + cut_frequency**2)
            return (-4 * J * (1 - np.cos(omega*t))
                    / (np.tanh(0.5*omega / temperature) * omega**2))

        # Calculate the analytical results by numerical integration
        expected = [0.5*np.exp(quad(_integrand, 0, np.inf, args=(t,))[0])
                    for t in times]

        H_sys = Qobj(np.zeros((2, 2)))
        if fake_timedep:
            H_sys = [H_sys]
        Q = sigmaz()
        initial_state = 0.5*Qobj(np.ones((2, 2)))
        projector = basis(2, 0) * basis(2, 1).dag()
        options = Options(nsteps=15_000, store_states=True)

        hsolver = HSolverDL(H_sys, Q, coupling_strength, temperature,
                            14, 2, cut_frequency,
                            bnd_cut_approx=bnd_cut_approx,
                            options=options)

        test = expect(hsolver.run(initial_state, times).states, projector)

        np.testing.assert_allclose(test, expected, atol=tol)


class TestFermionicHEOMSolver:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    @pytest.mark.parametrize(['fake_timedep'], [
        pytest.param(False, id="static"),
        pytest.param(True, id="timedep"),
    ])
    def test_steady_state_discrete_level_model(self, fake_timedep):
        """ Compare to discrete-level current analytics. """
        tol = 1e-3
        Gamma = 0.01  # coupling strength
        W = 1.  # cut-off
        T = 0.025851991  # temperature
        beta = 1. / T

        theta = 2.  # Bias
        mu_l = theta/2.
        mu_r = -theta/2.

        lmax = 10  # Pade cut-off

        def deltafun(j, k):
            if j == k:
                return 1.
            else:
                return 0.

        def Gamma_L_w(w):
            return Gamma * W**2 / ((w-mu_l)**2 + W**2)

        def Gamma_w(w, mu):
            return Gamma * W**2 / ((w-mu)**2 + W**2)

        def f(x):
            return 1 / (np.exp(x) + 1.)

        Alpha = np.zeros((2 * lmax, 2 * lmax))
        for j in range(2*lmax):
            for k in range(2*lmax):
                Alpha[j][k] = (
                    (deltafun(j, k + 1) + deltafun(j, k - 1))
                    / np.sqrt((2 * (j + 1) - 1) * (2 * (k + 1) - 1))
                )

        eigvalsA = eigvalsh(Alpha)

        eps = []
        for val in eigvalsA[0:lmax]:
            eps.append(-2 / val)

        AlphaP = np.zeros((2 * lmax - 1, 2 * lmax - 1))
        for j in range(2 * lmax - 1):
            for k in range(2 * lmax - 1):
                AlphaP[j][k] = (
                    (deltafun(j, k + 1) + deltafun(j, k - 1))
                    / np.sqrt((2 * (j + 1) + 1) * (2 * (k + 1) + 1))
                )

        eigvalsAP = eigvalsh(AlphaP)

        chi = []
        for val in eigvalsAP[0:lmax - 1]:
            chi.append(-2/val)

        eta_list = [
            0.5 * lmax * (2 * (lmax + 1) - 1) * (
                np.prod([chi[k]**2 - eps[j]**2 for k in range(lmax - 1)]) /
                np.prod([
                    eps[k]**2 - eps[j]**2 + deltafun(j, k) for k in range(lmax)
                ])
            )
            for j in range(lmax)
        ]

        kappa = [0] + eta_list
        epsilon = [0] + eps

        def f_approx(x):
            f = 0.5
            for ll in range(1, lmax + 1):
                f = f - 2 * kappa[ll] * x / (x**2 + epsilon[ll]**2)
            return f

        def C(sigma, mu):
            eta_0 = 0.5 * Gamma * W * f_approx(1.0j * beta * W)
            gamma_0 = W - sigma*1.0j*mu
            eta_list = [eta_0]
            gamma_list = [gamma_0]
            if lmax > 0:
                for ll in range(1, lmax + 1):
                    eta_list.append(
                        -1.0j * (kappa[ll] / beta) * Gamma * W**2
                        / (-(epsilon[ll]**2 / beta**2) + W**2)
                    )
                    gamma_list.append(epsilon[ll]/beta - sigma*1.0j*mu)
            return eta_list, gamma_list

        etapL, gampL = C(1.0, mu_l)
        etamL, gammL = C(-1.0, mu_l)

        etapR, gampR = C(1.0, mu_r)
        etamR, gammR = C(-1.0, mu_r)

        # heom simulation with above params (Pade)
        options = Options(
            nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14,
        )

        # Single fermion.
        d1 = destroy(2)

        # Site energy
        e1 = 1.

        H0 = e1 * d1.dag() * d1

        shape = H0.shape[0]
        dims = H0.dims

        if fake_timedep:
            H0 = [H0]

        # There are two leads, but we seperate the interaction into two terms,
        # labelled with \sigma=\pm such that there are 4 interaction operators
        # (See paper)
        Qops = [d1.dag(), d1, d1.dag(), d1]

        Kk = lmax + 1
        Ncc = 2  # For a single impurity we converge with Ncc = 2

        eta_list = [etapR, etamR, etapL, etamL]
        gamma_list = [gampR, gammR, gampL, gammL]
        Qops = [d1.dag(), d1, d1.dag(), d1]

        resultHEOM2 = FermionicHEOMSolver(
            H0, Qops,  eta_list, gamma_list, Ncc, options=options)

        rhossHP2, fullssP2 = resultHEOM2.steady_state()

        def get_aux_matrices(full, level, N_baths, Nk, N_cut, shape, dims):
            """
            Extracts the auxiliary matrices at a particular level
            from the full hierarchy ADOs.

            Parameters
            ----------
            full: ndarray
                A 2D array of the time evolution of the ADOs.

            level: int
                The level of the hierarchy to get the ADOs.

            N_cut: int
                The hierarchy cutoff.

            k: int
                The total number of exponentials used in each bath (assumed
                equal).

            N_baths: int
                The number of baths.

            shape : int
                the size of the ''system'' hilbert space

            dims : list
                the dimensions of the system hilbert space
            """
            # Note: Max N_cut is Nk*N_baths
            nstates, state2idx, idx2state = enr_state_dictionaries(
                [2] * (Nk * N_baths), N_cut,
            )  # _heom_state_dictionaries([Nc + 1] * (Nk), Nc)
            aux_indices = []

            aux_heom_indices = []
            for stateid in state2idx:
                if np.sum(stateid) == level:
                    aux_indices.append(state2idx[stateid])
                    aux_heom_indices.append(stateid)
            full = np.array(full)
            aux = []

            for i in aux_indices:
                qlist = [
                    Qobj(full[k, i, :].reshape(shape, shape).T, dims=dims)
                    for k in range(len(full))
                ]
                aux.append(qlist)
            return aux, aux_heom_indices, idx2state

        K = Kk

        aux_1_list, aux1_indices, idx2state = get_aux_matrices(
            [fullssP2], 1, 4, K, Ncc, shape, dims)
        aux_2_list, aux2_indices, idx2state = get_aux_matrices(
            [fullssP2], 2, 4, K, Ncc, shape, dims)

        d1 = destroy(2)  # Kk to 2*Kk
        currP = -1.0j * (
            sum([(d1 * aux_1_list[gg][0]).tr() for gg in range(Kk, 2 * Kk)]) -
            sum([(d1.dag() * aux_1_list[gg][0]).tr() for gg in range(Kk)])
        )

        def CurrFunc():
            def lamshift(w, mu):
                return (w-mu)*Gamma_w(w, mu)/(2*W)

            def integrand(w):
                return (
                    ((2 / (np.pi)) * Gamma_w(w, mu_l) * Gamma_w(w, mu_r) *
                        (f(beta * (w - mu_l)) - f(beta * (w - mu_r)))) /
                    ((Gamma_w(w, mu_l) + Gamma_w(w, mu_r))**2 + 4 *
                        (w - e1 - lamshift(w, mu_l) - lamshift(w, mu_r))**2)
                )

            def real_func(x):
                return np.real(integrand(x))

            def imag_func(x):
                return np.imag(integrand(x))

            # These integral bounds should be checked to be wide enough if the
            # parameters are changed
            a = -2
            b = 2
            real_integral = quad(real_func, a, b)
            imag_integral = quad(imag_func, a, b)

            return real_integral[0] + 1.0j * imag_integral[0]

        curr_ana = CurrFunc()
        np.testing.assert_allclose(curr_ana, -currP, atol=tol)

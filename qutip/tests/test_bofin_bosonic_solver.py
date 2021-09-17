"""
Tests for the Bosonic HEOM solvers.
"""

from math import factorial

import numpy as np
import pytest
from scipy.integrate import quad

from qutip import (
    Qobj, QobjEvo, sigmaz, sigmax, basis, expect, Options
)
from qutip.nonmarkov.bofin import (
    _convert_h_sys,
    _heom_state_dictionaries,
    add_at_idx,
    BosonicHEOMSolver,
    HSolverDL,
)


def test_add_at_idx():
    """
    Tests the function to add at hierarchy index.
    """
    seq = (2, 3, 4)
    assert add_at_idx(seq, 2, 1) == (2, 3, 5)

    seq = (2, 3, 4)
    assert add_at_idx(seq, 0, -1) == (1, 3, 4)


def test_state_dictionaries():
    """
    Tests the _heom_state_dictionaries.
    """
    kcut = 6
    N_cut = 4
    nhe, he2idx, idx2he = _heom_state_dictionaries(
            [N_cut + 1] * kcut, N_cut
        )

    total_nhe = int(
            factorial(N_cut + kcut)
            / (factorial(N_cut) * factorial(kcut))
        )
    assert nhe == total_nhe


def test_convert_h_sys():
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


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize(['bnd_cut_approx', 'tol', 'fake_timedep'], [
    pytest.param(True, 1e-4, False, id="bnd_cut_approx_static"),
    pytest.param(False,  1e-3, False, id="no_bnd_cut_approx_static"),
    pytest.param(True, 1e-4, True, id="bnd_cut_approx_timedep"),
    pytest.param(False,  1e-3, True, id="no_bnd_cut_approx_timedep"),
])
def test_pure_dephasing_model_HSolverDL(bnd_cut_approx, tol, fake_timedep):
    """
    HSolverDL: Compare with pure-dephasing analytical assert that the
    analytical result and HEOM produce the same time dephasing evoltion.
    """
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

    with pytest.warns(
        UserWarning,
        match=(
            "Two similar real and imag exponents have been collated"
            " automatically"
        ),
    ):
        hsolver = HSolverDL(H_sys, Q, coupling_strength, temperature,
                            14, 2, cut_frequency,
                            bnd_cut_approx=bnd_cut_approx,
                            options=options)

    test = expect(hsolver.run(initial_state, times).states, projector)

    np.testing.assert_allclose(test, expected, atol=tol)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize(['fake_timedep'], [
    pytest.param(False, id="static"),
    pytest.param(True, id="timedep"),
])
def test_pure_dephasing_model_BosonicHEOMSolver(fake_timedep):
    """
    BosonicHEOMSolver: Compare with pure-dephasing analytical assert that the
    analytical result and HEOM produce the same time dephasing evoltion.
    """
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

    with pytest.warns(
        UserWarning,
        match=(
            "Two similar real and imag exponents have been collated"
            " automatically"
        ),
    ):
        hsolver = BosonicHEOMSolver(
            H_sys, Q2, ckAR, ckAI, vkAR, vkAI,
            14, options=options,
        )

    test = expect(hsolver.run(initial_state, times).states, projector)

    np.testing.assert_allclose(test, expected, atol=tol)

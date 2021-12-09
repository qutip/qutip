# -*- coding: utf-8 -*-

"""
Test the Hierarchical Model Solver from qutip.nonmarkov.heom.
"""

import numpy as np
from scipy.integrate import quad
import pytest
import qutip
from qutip.nonmarkov.dlheom_solver import HSolverDL


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize(['renorm', 'bnd_cut_approx', 'stats', 'tol'], [
    pytest.param(True, True, True, 1e-4, id="renorm-bnd_cut_approx-stats"),
    pytest.param(True, False, False, 1e-3, id="renorm"),
    pytest.param(False, True, False, 1e-4, id="bnd_cut_approx"),
])
def test_pure_dephasing_model(renorm, bnd_cut_approx, stats, tol):
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
    expected = [
        0.5*np.exp(quad(_integrand, 0, np.inf, args=(t,), limit=5000)[0])
        for t in times
    ]

    H_sys = qutip.Qobj(np.zeros((2, 2)))
    Q = qutip.sigmaz()
    initial_state = 0.5*qutip.Qobj(np.ones((2, 2)))
    projector = qutip.basis(2, 0) * qutip.basis(2, 1).dag()
    options = qutip.Options(nsteps=15_000, store_states=True)
    hsolver = HSolverDL(H_sys, Q, coupling_strength, temperature,
                        20, 2, cut_frequency,
                        renorm=renorm, bnd_cut_approx=bnd_cut_approx,
                        options=options, stats=stats)
    test = qutip.expect(hsolver.run(initial_state, times).states, projector)
    if stats:
        assert hsolver.stats is not None
    else:
        assert hsolver.stats is None
    np.testing.assert_allclose(test, expected, atol=tol)


@pytest.mark.filterwarnings("ignore:zvode.*Excess work done:UserWarning")
def test_integration_error():
    cut_frequency = 0.05
    coupling_strength = 0.025
    temperature = 1 / 0.95

    H_sys = qutip.Qobj(np.zeros((2, 2)))
    Q = qutip.sigmaz()
    initial_state = 0.5 * qutip.Qobj(np.ones((2, 2)))
    options = qutip.Options(nsteps=10)
    hsolver = HSolverDL(
        H_sys, Q, coupling_strength, temperature, 20, 2, cut_frequency,
        options=options,
    )

    with pytest.raises(RuntimeError) as err:
        hsolver.run(initial_state, [0, 10])

    assert str(err.value) == (
        "HSolverDL ODE integration error. Try increasing the nsteps given"
        " in the HSolverDL options (which increases the allowed substeps"
        " in each step between times given in tlist)."
    )


def test_set_unset_stats():
    # Arbitrary system, just checking that stats can be unset by `configure`
    args = [qutip.qeye(2), qutip.sigmaz(),
            0.1, 0.1, 10, 1, 0.1]
    hsolver = HSolverDL(*args, stats=True)
    hsolver.run(qutip.basis(2, 0).proj(), [0, 1])
    assert hsolver.stats is not None
    hsolver.configure(*args, stats=False)
    assert hsolver.stats is None

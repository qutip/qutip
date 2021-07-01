"""
Tests for the Bosonic HEOM solvers.
"""
import numpy as np
from numpy.linalg import eigvalsh
from scipy.integrate import quad
from qutip import (Qobj, QobjEvo, sigmaz, sigmax, basis, expect, Options, destroy, basis)
from bofin.heom import add_at_idx
from bofin.heom import HSolverDL
from bofin.heom import BosonicHEOMSolver
from bofin.heom import _heom_state_dictionaries, _check_Hsys
import pytest
from math import sqrt, factorial

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


    #bosonic
    kcut = 6
    N_cut = 4
    nhe, he2idx, idx2he = _heom_state_dictionaries(
            [N_cut + 1] * kcut, N_cut
        )
        
    total_nhe = int(
            factorial(N_cut + kcut)
            / (factorial(N_cut) * factorial(kcut))
        )
    assert nhe, total_nhe
        
def test_check_H():
    """Tests the function for checking system Hamiltonian"""
    _check_Hsys(sigmax())
    _check_Hsys([sigmax(), sigmaz()])
    _check_Hsys([[sigmax(), np.sin], [sigmaz(), np.cos]])
    _check_Hsys([[sigmax(), np.sin], [sigmaz(), np.cos]])
    _check_Hsys(QobjEvo([sigmaz(), sigmax(), sigmaz()]))


    err_msg = r"Hamiltonian format is incorrect."

    with pytest.raises(RuntimeError, match=err_msg):
       _check_Hsys(sigmax().full())

    with pytest.raises(RuntimeError, match=err_msg):
       _check_Hsys([[1 , 0], [0, 1]])

    with pytest.raises(RuntimeError, match=err_msg):
       _check_Hsys([sigmax(), [[1 , 0], [0, 1]]])

    err_msg = r"Incorrect time dependent function for Hamiltonian."

    with pytest.raises(RuntimeError, match=err_msg):
       _check_Hsys([[sigmax(), [0, np.pi]]])

    with pytest.raises(RuntimeError, match=err_msg):
       _check_Hsys([[sigmax(), np.sin(0.5)]])


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize(['bnd_cut_approx', 'tol'], [
    pytest.param(True, 1e-4, id="bnd_cut_approx"),
    pytest.param(False,  1e-3, id="no_bnd_cut_approx"),  
])
def test_pure_dephasing_model_HSolverDL( bnd_cut_approx,  tol):
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


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_pure_dephasing_model_BosonicHEOMSolver():
    """
    BosonicHEOMSolver: Compare with pure-dephasing analytical assert that the
    analytical result and HEOM produce the same time dephasing evoltion.
    """
    tol=1e-3
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
    
    Nk=2
    ckAR = [ lam * gamma * (1/np.tan(gamma / (2 * T)))]
    ckAR.extend([(4 * lam * gamma * T *  2 * np.pi * k * T / (( 2 * np.pi * k * T)**2 - gamma**2)) for k in range(1,Nk+1)])
    vkAR = [gamma]
    vkAR.extend([2 * np.pi * k * T for k in range(1,Nk+1)])
    ckAI = [lam * gamma * (-1.0)]
    vkAI = [gamma]
    
    H_sys = Qobj(np.zeros((2, 2)))
    Q = sigmaz()
        
    NR = len(ckAR)
    NI = len(ckAI)
    Q2 = [Q for kk in range(NR+NI)]
    initial_state = 0.5*Qobj(np.ones((2, 2)))
    projector = basis(2, 0) * basis(2, 1).dag()
    options = Options(nsteps=15000, store_states=True)
    hsolver =  BosonicHEOMSolver(H_sys, Q2, ckAR, ckAI, vkAR, vkAI, 
                14, options=options)
    test = expect(hsolver.run(initial_state, times).states, projector)
 
    np.testing.assert_allclose(test, expected, atol=tol)

    
        
    

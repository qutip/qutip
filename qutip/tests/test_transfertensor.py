import pytest
import qutip as qt
import numpy as np
from qutip.nonmarkov.transfertensor import ttmsolve


def test_ttmsolve_jc_model():
    """
    Checks the output of ttmsolve using an example from Jaynes-Cumming model,
    which can also be found in the qutip-notebooks repository.
    """
    # Define Hamiltonian and states
    N, kappa = 3, 1.0
    a = qt.tensor(qt.qeye(2), qt.destroy(N))
    sm = qt.tensor(qt.sigmam(), qt.qeye(N))
    sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
    H = kappa * (a.dag() * sm + a * sm.dag())
    c_ops = [np.sqrt(kappa) * a]
    # identity superoperator
    Id = qt.tensor(qt.qeye(2), qt.qeye(N))
    E0 = qt.sprepost(Id, Id)
    # partial trace superoperator
    ptracesuper = qt.tensor_contract(E0, (1, N))
    # initial states
    rho0a = qt.ket2dm(qt.basis(2, 0))
    psi0c = qt.basis(N, 0)
    rho0c = qt.ket2dm(psi0c)
    rho0 = qt.tensor(rho0a, rho0c)
    superrho0cav = qt.sprepost(qt.tensor(qt.qeye(2), psi0c),
                               qt.tensor(qt.qeye(2), psi0c.dag()))

    # calculate exact solution using mesolve
    times = np.arange(0, 5, 0.1)
    exactsol = qt.mesolve(H, rho0, times, c_ops, [])
    exact_z = qt.expect(sz, exactsol.states)

    # dynamical map
    def dynmap(t):
        # reduced dynamical map for the qubit at time t
        Et = qt.mesolve(H, E0, [0., t], c_ops, []).states[-1]
        return ptracesuper * (Et * superrho0cav)

    # solve using transfer method
    learning_times = np.arange(0, 10.0, 0.1)
    learning_maps = [dynmap(t) for t in learning_times]
    ttmsol = ttmsolve(learning_maps, rho0a, times)
    ttm_z = qt.expect(qt.sigmaz(), ttmsol.states)
    # check that ttm result and exact solution are close
    assert np.allclose(ttm_z, exact_z, atol=0.01)

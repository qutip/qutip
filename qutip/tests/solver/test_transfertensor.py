import pytest
import qutip
import numpy as np
from qutip.solver.nonmarkov.transfertensor import ttmsolve


@pytest.mark.parametrize("call", [True, False])
def test_ttmsolve_jc_model(call):
    """
    Checks the output of ttmsolve using an example from Jaynes-Cumming model,
    which can also be found in the qutip-notebooks repository.
    """
    # Define Hamiltonian and states
    N, kappa, g = 3, 1.0, 10
    a = qutip.tensor(qutip.qeye(2), qutip.destroy(N))
    sm = qutip.tensor(qutip.sigmam(), qutip.qeye(N))
    sz = qutip.tensor(qutip.sigmaz(), qutip.qeye(N))
    H = g * (a.dag() * sm + a * sm.dag())
    c_ops = [np.sqrt(kappa) * a]
    # identity superoperator
    Id = qutip.tensor(qutip.qeye(2), qutip.qeye(N))
    E0 = qutip.sprepost(Id, Id)
    # partial trace superoperator
    ptracesuper = qutip.tensor_contract(E0, (1, N))
    # initial states
    rho0a = qutip.ket2dm(qutip.basis(2, 0))
    psi0c = qutip.basis(N, 0)
    rho0c = qutip.ket2dm(psi0c)
    rho0 = qutip.tensor(rho0a, rho0c)
    superrho0cav = qutip.sprepost(
        qutip.tensor(qutip.qeye(2), psi0c), qutip.tensor(qutip.qeye(2), psi0c.dag())
    )

    # calculate exact solution using mesolve
    times = np.arange(0, 5.0, 0.1)
    exactsol = qutip.mesolve(H, rho0, times, c_ops, e_ops=[sz])

    if not call:
        learning_times = np.arange(0, 2.0, 0.1)
        Et_list = qutip.mesolve(H, E0, learning_times, c_ops).states
        learning_maps = [ptracesuper @ Et @ superrho0cav for Et in Et_list]
    else:
        prop = qutip.Propagator(qutip.liouvillian(H, c_ops))
        def learning_maps(t):
            return ptracesuper @ prop(t) @ superrho0cav

    # solve using transfer method
    ttmsol = ttmsolve(learning_maps, rho0a, times,
                      e_ops=[qutip.sigmaz()], num_learning=21)

    # check that ttm result and exact solution are close in the learning times
    assert np.allclose(ttmsol.expect[0], exactsol.expect[0], atol=1e-5)

"""
This module provides exact solvers for a system-bath setup using the
reaction coordinate method.
"""

# Author: Neill Lambert, Anubhav Vardhan
# Contact: nwlambert@gmail.com

__all__ = ['rcsolve']

import warnings
import numpy as np
import scipy.sparse as sp
from numpy import matrix
from numpy import linalg
from .. import (
    spre, spost, sprepost, thermal_dm, tensor, destroy,
    qeye, mesolve, qeye_like, qzero_like
)


def rcsolve(Hsys, psi0, tlist, e_ops, Q, wc, alpha, N, w_th, sparse=False,
            options=None):
    """
    Function to solve for an open quantum system using the
    reaction coordinate (RC) model.

    Parameters
    ----------
    Hsys: Qobj
        The system hamiltonian.
    psi0: Qobj
        Initial state of the system.
    tlist: List.
        Time over which system evolves.
    e_ops: list of :class:`.Qobj` / callback function single
        Single operator or list of operators for which to evaluate
        expectation values.
    Q: Qobj
        The coupling between system and bath.
    wc: Float
        Cutoff frequency.
    alpha: Float
        Coupling strength.
    N: Integer
        Number of cavity fock states.
    w_th: Float
        Temperature.
    sparse: Boolean
        Optional argument to call the sparse eigenstates solver if needed.
    options : dict
        Options for the solver.

    Returns
    -------
    output: Result
        System evolution.
    """
    if psi0.isket:
        psi0 = psi0.proj()

    if options is None:
        options = {}

    dot_energy, dot_state = Hsys.eigenstates(sparse=sparse)
    deltaE = dot_energy[1] - dot_energy[0]
    if (w_th < deltaE / 2):
        warnings.warn("Given w_th might not provide accurate results")
    gamma = deltaE / (2 * np.pi * wc)
    wa = 2 * np.pi * gamma * wc  # reaction coordinate frequency
    g = np.sqrt(np.pi * wa * alpha / 2.0)  # reaction coordinate coupling
    nb = (1 / (np.exp(wa/w_th) - 1))

    # Reaction coordinate hamiltonian/operators
    a = tensor(destroy(N), qeye_like(Hsys))
    unit = tensor(qeye(N), qeye_like(Hsys))
    Q_exp = tensor(qeye(N), Q)
    Hsys_exp = tensor(qeye(N), Hsys)
    e_ops_exp = [tensor(qeye(N), kk) for kk in e_ops]

    na = a.dag() * a
    xa = a.dag() + a

    # decoupled Hamiltonian
    H0 = wa * a.dag() * a + Hsys_exp
    # interaction
    H1 = (g * (a.dag() + a) * Q_exp)
    H = H0 + H1
    PsipreEta = qzero_like(H)
    PsipreX = qzero_like(H)

    all_energy, all_state = H.eigenstates(sparse=sparse)
    Nmax = len(all_energy)
    for j in range(Nmax):
        for k in range(Nmax):
            A = xa.matrix_element(all_state[j].dag(), all_state[k])
            delE = (all_energy[j] - all_energy[k])
            if abs(A) > 0.0:
                if abs(delE) > 0.0:
                    eta = 0.5 * np.pi * gamma * delE * A
                    X = eta / np.tanh(delE / (2 * w_th))
                    proj = all_state[j] * all_state[k].dag()
                    PsipreX = PsipreX + X * proj
                    PsipreEta = PsipreEta + eta * proj
                else:
                    X = 0.5 * np.pi * gamma * A * 2 * w_th
                    PsipreX = PsipreX + X * all_state[j] * all_state[k].dag()

    A = a + a.dag()
    L = ((-spre(A * PsipreX)) + (sprepost(A, PsipreX))
         + (sprepost(PsipreX, A)) + (-spost(PsipreX * A))
         + (spre(A * PsipreEta)) + (sprepost(A, PsipreEta))
         + (-sprepost(PsipreEta, A)) + (-spost(PsipreEta * A)))

    # Setup the operators and the Hamiltonian and the master equation
    # and solve for time steps in tlist
    rho0 = (tensor(thermal_dm(N, nb), psi0))
    output = mesolve(H, rho0, tlist, [L], e_ops_exp, options=options)

    return output

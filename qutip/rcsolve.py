# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
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
from qutip import spre, spost, sprepost, thermal_dm, mesolve, Options
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis, qeye, dims


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
    e_ops: list of :class:`qutip.Qobj` / callback function single
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
    options : :class:`qutip.Options`
        With options for the solver.

    Returns
    -------
    output: Result
        System evolution.
    """
    if options is None:
        options = Options()

    dot_energy, dot_state = Hsys.eigenstates(sparse=sparse)
    deltaE = dot_energy[1] - dot_energy[0]
    if (w_th < deltaE/2):
        warnings.warn("Given w_th might not provide accurate results")
    gamma = deltaE / (2 * np.pi * wc)
    wa = 2 * np.pi * gamma * wc  # reaction coordinate frequency
    g = np.sqrt(np.pi * wa * alpha / 2.0)  # reaction coordinate coupling
    nb = (1 / (np.exp(wa/w_th) - 1))

    # Reaction coordinate hamiltonian/operators

    dimensions = dims(Q)
    a = tensor(destroy(N), qeye(dimensions[1]))
    unit = tensor(qeye(N), qeye(dimensions[1]))
    Nmax = N * dimensions[1][0]
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
    L = 0
    PsipreEta = 0
    PsipreX = 0

    all_energy, all_state = H.eigenstates(sparse=sparse)
    Apre = spre((a + a.dag()))
    Apost = spost(a + a.dag())
    for j in range(Nmax):
        for k in range(Nmax):
            A = xa.matrix_element(all_state[j].dag(), all_state[k])
            delE = (all_energy[j] - all_energy[k])
            if abs(A) > 0.0:
                if abs(delE) > 0.0:
                    X = (0.5 * np.pi * gamma*(all_energy[j] - all_energy[k])
                         * (np.cosh((all_energy[j] - all_energy[k]) /
                            (2 * w_th))
                         / (np.sinh((all_energy[j] - all_energy[k]) /
                            (2 * w_th)))) * A)
                    eta = (0.5 * np.pi * gamma *
                           (all_energy[j] - all_energy[k]) * A)
                    PsipreX = PsipreX + X * all_state[j] * all_state[k].dag()
                    PsipreEta = PsipreEta + (eta * all_state[j]
                                             * all_state[k].dag())
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

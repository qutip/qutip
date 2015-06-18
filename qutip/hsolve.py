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
hierachical method.
"""

# Author: Neill Lambert, Anubhav Vardhan
# Contact: nwlambert@gmail.com

__all__ = ['hsolve']

import numpy as np
import scipy.sparse as sp
import scipy.integrate
from copy import copy
from numpy import matrix
from numpy import linalg
from scipy.misc import factorial
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip import spre, spost, sprepost, Options, dims, qeye
from qutip import liouvillian, mat2vec, state_number_enumerate
from qutip import enr_state_dictionaries


def cot(x):
    """
    Calculate cotangent.
    Parameters
    ----------
    x: Float
        Angle.
    """
    return np.cos(x)/np.sin(x)


def hsolve(H, psi0, tlist, Q, gam, lam0, Nc, N, w_th, options=None):
    """
    Function to solve for an open quantum system using the
    hierarchy model.

    Parameters
    ----------
    H: Qobj
        The system hamiltonian.
    psi0: Qobj
        Initial state of the system.
    tlist: List.
        Time over which system evolves.
    Q: Qobj
        The coupling between system and bath.
    gam: Float
        Bath cutoff frequency.
    lam0: Float
        Coupling strength.
    Nc: Integer
        Cutoff parameter.
    N: Integer
        Number of matsubara terms.
    w_th: Float
        Temperature.
    options : :class:`qutip.Options`
        With options for the solver.

    Returns
    -------
    output: Result
        System evolution.
    """
    if options is None:
        options = Options()

    # Set up terms of the matsubara and tanimura boundaries

    # Parameters and hamiltonian
    hbar = 1.
    kb = 1.

    # Set by system
    dimensions = dims(H)
    Nsup = dimensions[0][0] * dimensions[0][0]
    unit = qeye(dimensions[0])

    # Ntot is the total number of ancillary elements in the hierarchy
    Ntot = int(round(factorial(Nc+N) / (factorial(Nc) * factorial(N))))
    c0 = (lam0 * gam * (cot(gam * hbar / (2. * kb * w_th)) - (1j))) / hbar
    LD1 = (-2. * spre(Q) * spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q))
    pref = ((2. * lam0 * kb * w_th / (gam * hbar)) - 1j * lam0) / hbar
    gj = 2 * np.pi * kb * w_th / hbar
    L12 = -pref * LD1 + (c0 / gam) * LD1

    for i1 in range(1, N):
        num = (4 * lam0 * gam * kb * w_th * i1 * gj/((i1 * gj)**2 - gam**2))
        ci = num / (hbar**2)
        L12 = L12 + (ci / gj) * LD1

    # Setup liouvillian

    L = liouvillian(H, [L12])
    Ltot = L.data
    unit = sp.eye(Ntot,format='csr')
    Lbig = sp.kron(unit, Ltot)
    rho0big1 = np.zeros((Nsup * Ntot), dtype=complex)

    # Prepare initial state:

    rhotemp = mat2vec(np.array(psi0.full(), dtype=complex))

    for idx, element in enumerate(rhotemp):
        rho0big1[idx] = element[0]
    
    nstates, state2idx, idx2state = enr_state_dictionaries([Nc+1]*(N), Nc)
    for nlabelt in state_number_enumerate([Nc+1]*(N), Nc):
        nlabel = list(nlabelt)
        ntotalcheck = 0
        for ncheck in range(N):
            ntotalcheck = ntotalcheck + nlabel[ncheck]
        current_pos = int(round(state2idx[tuple(nlabel)]))
        Ltemp = sp.lil_matrix((Ntot, Ntot))
        Ltemp[current_pos, current_pos] = 1
        Ltemp.tocsr()
        Lbig = Lbig + sp.kron(Ltemp, (-nlabel[0] * gam * spre(unit).data))

        for kcount in range(1, N):
            counts = -nlabel[kcount] * kcount * gj * spre(unit).data
            Lbig = Lbig + sp.kron(Ltemp, counts)

        for kcount in range(N):
            if nlabel[kcount] >= 1:
                # find the position of the neighbour
                nlabeltemp = copy(nlabel)
                nlabel[kcount] = nlabel[kcount] - 1
                current_pos2 = int(round(state2idx[tuple(nlabel)]))
                Ltemp = sp.lil_matrix((Ntot, Ntot))
                Ltemp[current_pos, current_pos2] = 1
                Ltemp.tocsr()
                # renormalized version:
                ci = (4 * lam0 * gam * kb * w_th * kcount
                      * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                if kcount == 0:
                    Lbig = Lbig + sp.kron(Ltemp, (-1j
                                          * (np.sqrt(nlabeltemp[kcount]
                                             / abs(c0)))
                                          * ((c0) * spre(Q).data
                                             - (np.conj(c0))
                                             * spost(Q).data)))
                if kcount > 0:
                    ci = (4 * lam0 * gam * kb * w_th * kcount
                          * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                    Lbig = Lbig + sp.kron(Ltemp, (-1j
                                          * (np.sqrt(nlabeltemp[kcount]
                                             / abs(ci)))
                                          * ((ci) * spre(Q).data
                                             - (np.conj(ci))
                                             * spost(Q).data)))
                nlabel = copy(nlabeltemp)

        for kcount in range(N):
            if ntotalcheck <= (Nc-1):
                nlabeltemp = copy(nlabel)
                nlabel[kcount] = nlabel[kcount] + 1
                current_pos3 = int(round(state2idx[tuple(nlabel)]))
            if current_pos3 <= (Ntot):
                Ltemp = sp.lil_matrix((Ntot, Ntot))
                Ltemp[current_pos, current_pos3] = 1
                Ltemp.tocsr()
            # renormalized
                if kcount == 0:
                    Lbig = Lbig + sp.kron(Ltemp, -1j
                                          * (np.sqrt((nlabeltemp[kcount]+1)
                                             * abs(c0)))
                                          * (spre(Q) - spost(Q)).data)
                if kcount > 0:
                    ci = (4 * lam0 * gam * kb * w_th * kcount
                          * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                    Lbig = Lbig + sp.kron(Ltemp, -1j
                                          * (np.sqrt((nlabeltemp[kcount]+1)
                                             * abs(ci)))
                                          * (spre(Q) - spost(Q)).data)
            nlabel = copy(nlabeltemp)

    output = []
    for element in rhotemp:
        output.append([])
    r = scipy.integrate.ode(cy_ode_rhs)
    Lbig2 = Lbig.tocsr()
    r.set_f_params(Lbig2.data, Lbig2.indices, Lbig2.indptr)
    r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step, max_step=options.max_step)

    r.set_initial_value(rho0big1, tlist[0])
    dt = tlist[1] - tlist[0]

    for t_idx, t in enumerate(tlist):
        r.integrate(r.t + dt)
        for idx, element in enumerate(rhotemp):
            output[idx].append(r.y[idx])

    return output

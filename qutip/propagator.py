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

__all__ = ['propagator', 'propagator_steadystate']

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from qutip.qobj import Qobj
from qutip.operators import qeye
from qutip.superoperator import (vec2mat, mat2vec)
from qutip.mesolve import mesolve, MeSolver
from qutip.sesolve import sesolve, SeSolver
from qutip.qobjevo_maker import qobjevo_maker
from qutip.states import basis
from qutip.solver import Options
from qutip.parallel import parallel_map, _default_kwargs
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar


def propagator(H, tlist, c_ops=[], args={}, options=None,
               unitary_mode='batch', parallel=False,
               progress_bar=None, _safe_mode=True,
               **kwargs):
    """
    Calculate the propagator U(t) for the density matrix or wave function such
    that :math:`\psi(t) = U(t)\psi(0)` or
    :math:`\\rho_{\mathrm vec}(t) = U(t) \\rho_{\mathrm vec}(0)`
    where :math:`\\rho_{\mathrm vec}` is the vector representation of the
    density matrix.

    Parameters
    ----------
    H : qobj or list
        Hamiltonian as a Qobj instance of a nested list of Qobjs and
        coefficients in the list-string or list-function format for
        time-dependent Hamiltonians (see description in :func:`qutip.mesolve`).

    tlist : float or array-like
        Time or list of times for which to evaluate the propagator.

    c_ops : list
        List of qobj collapse operators.

    args : list/array/dictionary
        Parameters to callback functions for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.Options`
        with options for the ODE solver.

    unitary_mode = str ('batch', 'single')
        Solve all basis vectors simulaneously ('batch') or individually
        ('single'). Deprecated/ignored, single if parallel, batch otherwise.

    parallel : bool {False, True}
        Run the propagator in parallel mode. This will override the
        unitary_mode settings if set to True.

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation. By default no progress bar
        is used, and if set to True a TextProgressBar will be used.

    Returns
    -------
     a : qobj
        Instance representing the propagator :math:`U(t)`.

    """
    kw = _default_kwargs()
    num_cpus = kwargs['num_cpus'] if 'num_cpus' in kwargs else kw['num_cpus']

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    if isinstance(tlist, (int, float, np.integer, np.floating)):
        tlist = [0, tlist]
        one_time = True
    else:
        one_time = False

    H_td = qobjevo_maker(H, args, tlist=tlist)

    if len(c_ops) == 0 and not H_td.issuper:
        # calculate propagator for the wave function
        N = H_td.shape[0]
        dims = H_td.dims
        if parallel:
            system = SeSolver(H_td, args, tlist=tlist, options=options)
            output = parallel_map(_parallel_sesolve, range(N),
                                  task_args=(N, system, tlist),
                                  progress_bar=progress_bar, num_cpus=num_cpus)

            u = np.zeros([N, N, len(tlist)], dtype=complex)
            for n in range(N):
                for k, _ in enumerate(tlist):
                    u[:, n, k] = output[n].states[k]
            Us = [Qobj(u[:,:,t], dims=dims) for t in range(len(tlist))]

        else:
            Us = sesolve(H_td, qeye(dims[0]), tlist, [], args, options,
                         progress_bar=progress_bar, _safe_mode=False).states

    elif len(c_ops) or H_td.issuper:
        # calculate the propagator for the vector representation of the
        # density matrix (a superoperator propagator)
        if H_td.issuper:
            N = H_td.shape[0]
            sqrt_N = int(np.sqrt(N))
            dims = H_td.dims
        else:
            N = H_td.shape[0]**2
            sqrt_N = H_td.shape[0]
            dims = [H_td.dims, H_td.dims]

        if parallel:
            u = np.zeros([N, N, len(tlist)], dtype=complex)
            system = MeSolver(H_td, c_ops, args, tlist=tlist, options=options)
            output = parallel_map(_parallel_mesolve, range(N),
                                  task_args=(sqrt_N, system, tlist),
                                  progress_bar=progress_bar, num_cpus=num_cpus)
            for n in range(N):
                for k, t in enumerate(tlist):
                    u[:, n, k] = mat2vec(output[n].states[k]).T
            Us = [Qobj(u[:,:,t], dims=dims) for t in range(len(tlist))]
        else:
            rho0 = qeye(N)
            rho0.dims = dims
            Us = mesolve(H_td, rho0, tlist, c_ops, [], args,
                         options, progress_bar=progress_bar).states


    return Us if not one_time else Us[-1]


def propagator_steadystate(U):
    """Find the steady state for successive applications of the propagator
    :math:`U`.

    Parameters
    ----------
    U : qobj
        Operator representing the propagator.

    Returns
    -------
    a : qobj
        Instance representing the steady-state density matrix.

    """

    evals, evecs = la.eig(U.full())

    shifted_vals = np.abs(evals - 1.0)
    ev_idx = np.argmin(shifted_vals)
    ev_min = shifted_vals[ev_idx]
    evecs = evecs.T
    rho = Qobj(vec2mat(evecs[ev_idx]), dims=U.dims[0])
    rho = rho * (1.0 / rho.tr())
    rho = 0.5 * (rho + rho.dag())  # make sure rho is herm
    rho.isherm = True
    return rho


def _parallel_sesolve(n, N, system, tlist):
    psi0 = basis(N, n)
    return system.run(tlist, psi0, outtype="dense")

def _parallel_mesolve(n, N, system, tlist):
    col_idx, row_idx = np.unravel_index(n, (N, N))
    rho0 = Qobj(sp.csr_matrix(([1], ([row_idx], [col_idx])),
                              shape=(N,N), dtype=complex))
    return system.run(tlist, rho0, outtype="dense")

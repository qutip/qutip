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

import functools
import numbers
import types
import numpy as np

from .. import (
    Qobj, qeye, unstack_columns, stack_columns, basis, projection,
)
from ..core import data as _data
# from ._rhs_generate import rhs_clear, td_format_check
from .mesolve import *
from .sesolve import *
from .options import SolverOptions
from .parallel import parallel_map
from ..ui.progressbar import BaseProgressBar, TextProgressBar


def propagator(H, t, c_op_list=[], args={}, options=None,
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

    t : float or array-like
        Time or list of times for which to evaluate the propagator.

    c_op_list : list
        List of qobj collapse operators.

    args : list/array/dictionary
        Parameters to callback functions for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.SolverOptions`
        with options for the ODE solver.

    unitary_mode = str ('batch', 'single')
        Solve all basis vectors simulaneously ('batch') or individually
        ('single').

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
    # TODO: correct for proper ammout
    num_cpus = kwargs.get('num_cpus', 1)

    if options is None:
        options = SolverOptions()
    progress_bar = get_progess_bar(options['progress_bar'])

    if isinstance(t, numbers.Real):
        tlist = [0, t]
    else:
        tlist = t

    unitary_mode = 'single'
    if isinstance(H, (types.FunctionType, types.BuiltinFunctionType,
                      functools.partial)):
        H0 = H(0.0, args)
    elif isinstance(H, list):
        H0 = H[0][0] if isinstance(H[0], list) else H[0]
    else:
        H0 = H

    N = H0.shape[0]

    if len(c_op_list) == 0 and H0.isoper:
        # calculate propagator for the wave function
        dims = H0.dims
        if parallel:
            u = np.zeros([N, N, len(tlist)], dtype=complex)
            output = parallel_map(_parallel_sesolve, range(N),
                                  task_args=(N, H, tlist, args, options),
                                  progress_bar=progress_bar, num_cpus=num_cpus)
            for n in range(N):
                for k, t in enumerate(tlist):
                    u[:, n, k] = output[n].states[k].full().T
            output = [Qobj(u[:, :, k], dims=dims) for k in range(len(tlist))]
        else:
            output = sesolve(H, qeye(dims[0]), tlist, [], args, options,
                             _safe_mode=False).states

    else:
        # calculate the propagator for the vector representation of the
        # density matrix (a superoperator propagator)
        if H0.issuper:
            sqrt_N = int(np.sqrt(N))
            dims = H0.dims
        else:
            dims = [H0.dims, H0.dims]
            sqrt_N = N
            N = N*N

        if parallel:
            u = np.zeros([N, N, len(tlist)], dtype=complex)
            output = parallel_map(_parallel_mesolve, range(N * N),
                                  task_args=(
                                      sqrt_N, H, tlist, c_op_list, args, options),
                                  progress_bar=progress_bar, num_cpus=num_cpus)
            for n in range(N * N):
                for k, state in enumerate(output[n].states):
                    u[:, n, k] = stack_columns(state.data).to_array()[:, 0]
            output = [Qobj(u[:, :, k], dims=dims) for k in range(len(tlist))]

        else:
            rho0 = qeye([sqrt_N, sqrt_N])
            output = mesolve(H, rho0, tlist, [], args, options).states

    return output[-1] if len(tlist) == 2 else output


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
    evals, estates = U.eigenstates()
    shifted_vals = np.abs(evals - 1.0)
    ev_idx = np.argmin(shifted_vals)
    rho_data = unstack_columns(estates[ev_idx].data)
    rho_data = _data.mul(rho_data, 0.5 / _data.trace(rho_data))
    return Qobj(_data.add(rho_data, _data.adjoint(rho_data)),
                dims=U.dims[0],
                type='oper',
                isherm=True,
                copy=False)


def _parallel_sesolve(n, N, H, tlist, args, options):
    psi0 = basis(N, n)
    output = sesolve(H, psi0, tlist, [], args, options, _safe_mode=False)
    return output

def _parallel_mesolve(n, N, H, tlist, c_op_list, args, options):
    col_idx, row_idx = np.unravel_index(n, (N, N))
    rho0 = projection(N, row_idx, col_idx)
    output = mesolve(H, rho0, tlist, c_op_list, [], args, options,
                     _safe_mode=False)
    return output

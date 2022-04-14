__all__ = ['propagator', 'propagator_steadystate']

import functools
import numbers
import types

import numpy as np
import scipy.sparse as sp

from .. import (
    Qobj, tensor, qeye, unstack_columns, stack_columns, basis, projection,
    QobjEvo
)
from ..core import data as _data
from .mesolve import mesolve
from .sesolve import sesolve
from .solver import SolverOptions, _solver_safety_check
from ..parallel import parallel_map, _default_kwargs
from ..ui.progressbar import BaseProgressBar, TextProgressBar


def propagator(H, t, c_op_list=[], args={}, options=None,
               unitary_mode='batch', parallel=False,
               progress_bar=None, _safe_mode=True,
               **kwargs):
    r"""
    Calculate the propagator U(t) for the density matrix or wave function such
    that :math:`\psi(t) = U(t)\psi(0)` or
    :math:`\rho_{\mathrm vec}(t) = U(t) \rho_{\mathrm vec}(0)`
    where :math:`\rho_{\mathrm vec}` is the vector representation of the
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

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    if options is None:
        options = SolverOptions()

    if isinstance(t, numbers.Real):
        tlist = [0, t]
    else:
        tlist = t

    if _safe_mode:
        _solver_safety_check(H, None, c_ops=c_op_list, e_ops=[], args=args)

    if isinstance(H, (types.FunctionType, types.BuiltinFunctionType,
                      functools.partial)):
        H0 = H(0.0, args)
        if unitary_mode == 'batch':
            # batch don't work with function Hamiltonian
            unitary_mode = 'single'
    elif isinstance(H, list):
        H0 = H[0][0] if isinstance(H[0], list) else H[0]
    else:
        H0 = H

    if len(c_op_list) == 0 and H0.isoper:
        # calculate propagator for the wave function

        N = H0.shape[0]
        dims = H0.dims

        if parallel:
            unitary_mode = 'single'
            u = np.zeros([N, N, len(tlist)], dtype=complex)
            output = parallel_map(_parallel_sesolve, range(N),
                                  task_args=(N, H, tlist, args, options),
                                  progress_bar=progress_bar, num_cpus=num_cpus)
            for n in range(N):
                for k, t in enumerate(tlist):
                    u[:, n, k] = output[n].states[k].full().T
        else:
            if unitary_mode == 'single':
                output = sesolve(H, qeye(dims[0]), tlist, [], args, options,
                                 _safe_mode=False)
                return output.states[-1] if len(tlist) == 2 else output.states

            elif unitary_mode == 'batch':
                u = np.zeros(len(tlist), dtype=object)
                rows_ = np.array([(N+1)*m for m in range(N)])
                cols_ = np.zeros_like(rows_)
                data_ = np.ones_like(rows_, dtype=complex)
                psi0 = Qobj(sp.coo_matrix((data_, (rows_, cols_))).tocsr())
                H = QobjEvo(H, args=args, tlist=tlist)
                H2 = tensor(qeye(N), H)

                options['normalize_output'] = False
                output = sesolve(H2, psi0, tlist, [],
                                 args=args, options=options,
                                 _safe_mode=False)
                for k, state in enumerate(output.states):
                    out = unstack_columns(state.data, (N, N)).to_array()
                    out /= np.linalg.norm(out, axis=1)
                    u[k] = _data.create(out)
            else:
                raise ValueError('Invalid unitary mode.')

    elif len(c_op_list) == 0 and H0.issuper:
        # calculate the propagator for the vector representation of the
        # density matrix (a superoperator propagator)
        unitary_mode = 'single'
        N = H0.shape[0]
        sqrt_N = int(np.sqrt(N))
        dims = H0.dims

        u = np.zeros([N, N, len(tlist)], dtype=complex)

        if parallel:
            output = parallel_map(_parallel_mesolve, range(N),
                                  task_args=(
                                      sqrt_N, H, tlist, c_op_list, args,
                                      options),
                                  task_kwargs={"dims": H0.dims[0]},
                                  progress_bar=progress_bar, num_cpus=num_cpus)
            for n in range(N):
                for k, state in enumerate(output[n].states):
                    u[:, n, k] = stack_columns(state.data).to_array()[:, 0]
        else:
            rho0 = qeye(dims[0])
            output = mesolve(H, rho0, tlist, [], args=args, options=options,
                             _safe_mode=False)
            return output.states[-1] if len(tlist) == 2 else output.states

    else:
        # calculate the propagator for the vector representation of the
        # density matrix (a superoperator propagator)
        unitary_mode = 'single'
        N = H0.shape[0]
        dims = [H0.dims, H0.dims]

        u = np.zeros([N * N, N * N, len(tlist)], dtype=complex)

        if parallel:
            output = parallel_map(_parallel_mesolve, range(N * N),
                                  task_args=(
                                      N, H, tlist, c_op_list, args, options),
                                  task_kwargs={"dims": H0.dims},
                                  progress_bar=progress_bar, num_cpus=num_cpus)
            for n in range(N * N):
                for k, state in enumerate(output[n].states):
                    u[:, n, k] = stack_columns(state.data).to_array()[:, 0]
        else:
            progress_bar.start(N * N)
            for n in range(N * N):
                progress_bar.update(n)
                col_idx, row_idx = np.unravel_index(n, (N, N))
                rho0 = projection(N, row_idx, col_idx)
                rho0.dims = H0.dims
                output = mesolve(
                    H, rho0, tlist, c_ops=c_op_list, args=args,
                    options=options, _safe_mode=False)
                for k, t in enumerate(tlist):
                    u[:, n, k] = stack_columns(output.states[k].data).to_array()[:, 0]
            progress_bar.finished()

    if len(tlist) == 2:
        data = u[-1] if unitary_mode == 'batch' else u[:, :, 1]
        return Qobj(data, dims=dims)

    out = np.empty((len(tlist),), dtype=object)
    if unitary_mode == 'batch':
        out[:] = [Qobj(u[k], dims=dims) for k in range(len(tlist))]
    else:
        out[:] = [Qobj(u[:, :, k], dims=dims) for k in range(len(tlist))]
    return out


def _get_min_and_index(lst):
    """
    Private function for obtaining min and max indicies.
    """
    minval, minidx = lst[0], 0
    for i, v in enumerate(lst[1:]):
        if v < minval:
            minval, minidx = v, i + 1
    return minval, minidx


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


def _parallel_mesolve(n, N, H, tlist, c_op_list, args, options, dims=None):
    col_idx, row_idx = np.unravel_index(n, (N, N))
    rho0 = projection(N, row_idx, col_idx)
    rho0.dims = dims
    output = mesolve(
        H, rho0, tlist, c_ops=c_op_list, args=args, options=options,
        _safe_mode=False)
    return output

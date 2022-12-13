"""
This module provides solvers for the unitary Schrodinger equation.
"""

__all__ = ['sesolve']

import numpy as np
import scipy.integrate
from scipy.linalg import norm as la_norm
from .. import Qobj, QobjEvo
from ..core import data as _data
from .solver import Result, SolverOptions, solver_safe, SolverSystem
from ..ui.progressbar import BaseProgressBar, TextProgressBar


def sesolve(H, psi0, tlist, e_ops=None, args=None, options=None,
            progress_bar=None, _safe_mode=True):
    """
    Schrödinger equation evolution of a state vector or unitary matrix for a
    given Hamiltonian.

    Evolve the state vector (``psi0``) using a given Hamiltonian (``H``), by
    integrating the set of ordinary differential equations that define the
    system. Alternatively evolve a unitary matrix in solving the Schrodinger
    operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (``tlist``), or the expectation values of the supplied operators
    (``e_ops``). If ``e_ops`` is a callback function, it is invoked for each
    time in ``tlist`` with time and the state as arguments, and the function
    does not use any return values. ``e_ops`` cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------

    H : :class:`~qutip.Qobj`, :class:`~qutip.QobjEvo`, list, or callable
        System Hamiltonian as a :obj:`~qutip.Qobj` , list of
        :obj:`~qutip.Qobj` and coefficient, :obj:`~qutip.QobjEvo`,
        or a callback function for time-dependent Hamiltonians. List format
        and options can be found in QobjEvo's description.

    psi0 : :class:`~qutip.Qobj`
        Initial state vector (ket) or initial unitary operator ``psi0 = U``.

    tlist : array_like of float
        List of times for :math:`t`.

    e_ops : None / list / callback function, optional
        A list of operators as `Qobj` and/or callable functions (can be mixed)
        or a single callable function. For callable functions, they are called
        as ``f(t, state)`` and return the expectation value. A single
        callback's expectation value can be any type, but a callback part of a
        list must return a number as the expectation value. For operators, the
        result's expect will be computed by :func:`qutip.expect` when the state
        is a ``ket``. For operator evolution, the overlap is computed by: ::

            (e_ops[i].dag() * op(t)).tr()

    args : dict, optional
        Dictionary of scope parameters for time-dependent Hamiltonians.

    options : None / :class:`qutip.SolverOptions`, optional
        Options for the ODE solver.

    progress_bar : :obj:`~BaseProgressBar`, optional
        Optional instance of :obj:`~BaseProgressBar`, or a subclass thereof,
        for showing the progress of the simulation.

    Returns
    -------

    output: :class:`~qutip.solver.Result`
        An instance of the class :class:`~qutip.solver.Options`, which
        contains either an array of expectation values for the times
        specified by ``tlist``, or an array or state vectors
        corresponding to the times in ``tlist`` (if ``e_ops`` is an empty
        list), or nothing if a callback function was given inplace of
        operators for which to calculate the expectation values.
    """
    if e_ops is None:
        e_ops = []
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    elif isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    if progress_bar is True:
        progress_bar = TextProgressBar()

    if not (psi0.isket or psi0.isunitary):
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")

    if options is None:
        options = SolverOptions()
    """
    if options.rhs_reuse and not isinstance(H, SolverSystem):
        # TODO: deprecate when going to class based solver.
        if "sesolve" in solver_safe:
            H = solver_safe["sesolve"]
        else:
            pass
            # raise Exception("Could not find the Hamiltonian to reuse.")
    """

    if args is None:
        args = {}

    if isinstance(H, SolverSystem):
        ss = H
    elif isinstance(H, (list, Qobj, QobjEvo)):
        ss = _sesolve_QobjEvo(H, tlist, args, options)
    elif callable(H):
        ss = _sesolve_func_td(H, args, options)
    else:
        raise TypeError(f"Invalid H: {H!r}")

    func, ode_args = ss.makefunc(ss, psi0, args, e_ops, options)

    if _safe_mode:
        v = psi0.full().ravel('F')
        func(0., v, *ode_args)[:, 0] + v

    res = _generic_ode_solve(func, ode_args, psi0, tlist, e_ops, options,
                             progress_bar, dims=psi0.dims)
    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#
def _sesolve_QobjEvo(H, tlist, args, opt):
    """
    Prepare the system for the solver, H can be an QobjEvo.
    """
    H_td = -1.0j * QobjEvo(H, args, tlist=tlist)

    ss = SolverSystem()
    ss.H = H_td
    ss.makefunc = _qobjevo_set
    solver_safe["sesolve"] = ss
    return ss


def _wrap_matmul(t, state, cqobj, oper):
    state = _data.dense.fast_from_numpy(state)
    if oper:
        state = _data.column_unstack_dense(state, cqobj.shape[1], inplace=True)
    out = cqobj.matmul_data(t, state)
    if oper:
        out = _data.column_stack_dense(out, inplace=True)
    return out.as_ndarray()


def _qobjevo_set(HS, psi, args, e_ops, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args)
    if psi.isket or psi.isunitary:
        return _wrap_matmul, (H_td, psi.isunitary)
    raise TypeError("The unitary solver requires psi0 to be"
                    " a ket as initial state"
                    " or a unitary as initial operator.")


# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians.
#
def _sesolve_func_td(H_func, args, opt):
    """
    Prepare the system for the solver, H is a function.
    """
    ss = SolverSystem()
    ss.H = H_func
    ss.makefunc = _Hfunc_set
    solver_safe["sesolve"] = ss
    return ss


def _Hfunc_set(HS, psi, args, e_ops, opt):
    """
    From the system, get the ode function and args
    """
    return _sesolve_rhs_func, (HS.H, args, psi.isunitary, False)


# -----------------------------------------------------------------------------
# evaluate dU(t)/dt according to the schrodinger equation
#
def _sesolve_rhs_func(t, y, H_func, args, oper, with_state):
    H = H_func(t, y, args) if with_state else H_func(t, args)
    if isinstance(H, Qobj):
        H = H.data
    y = _data.dense.fast_from_numpy(y)
    if oper:
        ym = _data.column_unstack_dense(y, H.shape[1], inplace=True)
        out = _data.matmul(H, ym, scale=-1j)
        return _data.column_stack_dense(out, inplace=True).as_ndarray()
    return _data.matmul(H, y, scale=-1j, dtype=_data.Dense).as_ndarray()


# -----------------------------------------------------------------------------
# Solve an ODE for func.
# Calculate the required expectation values or invoke callback
# function at each time step.
def _generic_ode_solve(func, ode_args, psi0, tlist, e_ops, opt,
                       progress_bar, dims=None):
    """
    Internal function for solving ODEs.
    """
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This function is made similar to mesolve's one for futur merging in a
    # solver class
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # prepare output array
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "sesolve"
    output.times = tlist

    if psi0.isunitary:
        initial_vector = psi0.full().ravel('F')
        oper_evo = True
    elif psi0.isket:
        initial_vector = psi0.full().ravel()
        oper_evo = False

    r = scipy.integrate.ode(func)
    r.set_integrator('zvode', method=opt['method'], order=opt['order'],
                     atol=opt['atol'], rtol=opt['rtol'], nsteps=opt['nsteps'],
                     first_step=opt['first_step'], min_step=opt['min_step'],
                     max_step=opt['max_step'])
    if ode_args:
        r.set_f_params(*ode_args)
    r.set_initial_value(initial_vector, tlist[0])

    e_ops_data = []
    output.expect = []
    if callable(e_ops):
        n_expt_op = 0
        expt_callback = True
        output.num_expect = 1
    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False
        output.num_expect = n_expt_op
        if n_expt_op == 0:
            # fallback on storing states
            opt['store_states'] = True
        else:
            for op in e_ops:
                if not isinstance(op, Qobj) and callable(op):
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
                    continue
                if op.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
        if oper_evo:
            for e in e_ops:
                if not isinstance(e, Qobj):
                    e_ops_data.append(e)
                elif e.dims[1] != psi0.dims[0]:
                    raise TypeError(f"e_ops dims ({e.dims}) are not compatible"
                                    f" with the state's ({psi0.dims})")
                else:
                    e_ops_data.append(e.dag().data)
        else:
            for e in e_ops:
                if not isinstance(e, Qobj):
                    e_ops_data.append(e)
                elif e.dims[1] != psi0.dims[0]:
                    raise TypeError(f"e_ops dims ({e.dims}) are not compatible"
                                    f" with the state's ({psi0.dims})")
                else:
                    e_ops_data.append(e.data)
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt['store_states']:
        output.states = []

    if oper_evo:
        def get_curr_state_data(r):
            return _data.column_unstack_dense(_data.dense.fast_from_numpy(r.y),
                                              psi0.shape[0], inplace=True)
    else:
        def get_curr_state_data(r):
            return _data.dense.fast_from_numpy(r.y)

    #
    # start evolution
    #
    dt = np.diff(tlist)
    cdata = None
    progress_bar.start(n_tsteps)
    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)
        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")
        # get the current state / oper data if needed
        if (
            opt['store_states']
            or opt['normalize_output']
            or n_expt_op > 0
            or expt_callback
        ):
            cdata = get_curr_state_data(r)

        if opt['normalize_output']:
            # normalize per column
            if oper_evo:
                cdata_nd = cdata.as_ndarray()
                cdata_nd /= la_norm(cdata_nd, axis=0)
                # Don't do this in place, because we use it later.
                initial = _data.column_stack_dense(cdata, inplace=False)
                r.set_initial_value(initial.as_ndarray(), r.t)
            else:
                norm = _data.norm.l2(cdata)
                if abs(norm - 1) > 1e-12:
                    # only reset the solver if state changed
                    cdata = _data.mul(cdata, 1/norm)
                    r.set_initial_value(cdata.as_ndarray(), r.t)
                else:
                    r._y = cdata.as_ndarray()

        if opt['store_states']:
            output.states.append(Qobj(cdata,
                                      dims=dims, type=psi0.type))

        if expt_callback:
            # use callback method
            output.expect.append(e_ops(t, Qobj(cdata,
                                               dims=dims, type=psi0.type)))

        for m in range(n_expt_op):
            if not isinstance(e_ops[m], Qobj) and callable(e_ops[m]):
                output.expect[m][t_idx] = e_ops[m](t, Qobj(cdata, dims=dims))
            else:
                val = _data.expect(e_ops_data[m], cdata)
                if e_ops[m].isherm:
                    val = val.real
                output.expect[m][t_idx] = val

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if opt['store_final_state']:
        cdata = get_curr_state_data(r)
        if opt['normalize_output']:
            cdata = cdata.as_ndarray()
            cdata /= la_norm(cdata, axis=0)
        output.final_state = Qobj(cdata, dims=dims)

    return output

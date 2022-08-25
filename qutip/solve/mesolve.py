"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['mesolve']

import numpy as np
import scipy.integrate
from .. import (
    Qobj, QobjEvo, isket, issuper, spre, spost, liouvillian,
    lindblad_dissipator, ket2dm,
)
from ..core import data as _data
from .solver import SolverOptions, Result, solver_safe, SolverSystem
from .sesolve import sesolve
from ..ui.progressbar import BaseProgressBar, TextProgressBar


# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, args=None, options=None,
            progress_bar=None, _safe_mode=True):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian or Liouvillian (`H`) and an optional set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system. In the absence of collapse operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows the solution of master equations that are not in standard
    Lindblad form.

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be a specified in a
    nested-list format where each element in the list is a list of length 2,
    containing an operator (:class:`qutip.qobj`) at the first element and where
    the second element is either a string (*list string format*), a callback
    function (*list callback format*) that evaluates to the time-dependent
    coefficient for the corresponding operator, or a NumPy array (*list
    array format*) which specifies the value of the coefficient to the
    corresponding operator for each value of t in `tlist`.

    Alternatively, `H` (but not `c_ops`) can be a callback function with the
    signature `f(t, args) -> Qobj` (*callback format*), which can return the
    Hamiltonian or Liouvillian superoperator at any point in time.  If the
    equation cannot be put in standard Lindblad form, then this time-dependence
    format must be used.

    *Examples*

        H = [[H0, 'sin(w*t)'], [H1, 'sin(2*w*t)']]

        H = [[H0, f0_t], [H1, f1_t]]

        where f0_t and f1_t are python functions with signature f_t(t, args).

        H = [[H0, np.sin(w*tlist)], [H1, np.sin(2*w*tlist)]]

    In the *list string format* and *list callback format*, the string
    expression and the callback function must evaluate to a real or complex
    number (coefficient for the corresponding operator).

    In all cases of time-dependent operators, `args` is a dictionary of
    parameters that is used when evaluating operators. It is passed to the
    callback functions as their second argument.

    **Additional options**

    Additional options to mesolve can be set via the `options` argument, which
    should be an instance of :class:`qutip.solver.SolverOptions`. Many ODE
    integration options can be set this way, and the `store_states` and
    `store_final_state` options can be used to store states even though
    expectation values are requested via the `e_ops` argument.

    .. note::

        If an element in the list-specification of the Hamiltonian or
        the list of collapse operators are in superoperator form it will be
        added to the total Liouvillian of the problem without further
        transformation. This allows for using mesolve for solving master
        equations that are not in standard Lindblad form.

    .. note::

        On using callback functions: mesolve transforms all :class:`qutip.Qobj`
        objects to sparse matrices before handing the problem to the integrator
        function. In order for your callback function to work correctly, pass
        all :class:`qutip.Qobj` objects that are used in constructing the
        Hamiltonian via `args`. mesolve will check for :class:`qutip.Qobj` in
        `args` and handle the conversion to sparse matrices. All other
        :class:`qutip.Qobj` objects that are not passed via `args` will be
        passed on to the integrator in scipy which will raise a NotImplemented
        exception.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian, or a callback function for time-dependent
        Hamiltonians, or alternatively a system Liouvillian.

    rho0 : :class:`qutip.Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : None / list of :class:`qutip.Qobj`
        single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators.

    e_ops : None / list / callback function, optional
        A list of operators as `Qobj` and/or callable functions (can be mixed)
        or a single callable function. For operators, the result's expect will
        be  computed by :func:`qutip.expect`. For callable functions, they are
        called as ``f(t, state)`` and return the expectation value.
        A single callback's expectation value can be any type, but a callback
        part of a list must return a number as the expectation value.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : None / :class:`qutip.SolverOptions`
        with options for the solver.

    progress_bar : None / BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        either an *array* `result.expect` of expectation values for the times
        specified by `tlist`, or an *array* `result.states` of state vectors or
        density matrices corresponding to the times in `tlist` [if `e_ops` is
        an empty list], or nothing if a callback function was given in place of
        operators for which to calculate the expectation values.

    """
    if c_ops is None:
        c_ops = []
    if isinstance(c_ops, (Qobj, QobjEvo)):
        c_ops = [c_ops]

    if e_ops is None:
        e_ops = []
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    if progress_bar is True:
        progress_bar = TextProgressBar()

    # check if rho0 is a superoperator, in which case e_ops argument should
    # be empty, i.e., e_ops = []
    # TODO: e_ops for superoperator
    if rho0.issuper and not e_ops == []:
        raise TypeError("Must have e_ops = [] when initial condition rho0 is"
                        " a superoperator.")

    if options is None:
        options = SolverOptions()
    if False and not isinstance(H, SolverSystem):
        # TODO: deprecate when going to class based solver.
        if "mesolve" in solver_safe:
            # print(" ")
            H = solver_safe["mesolve"]
        else:
            pass
            # raise Exception("Could not find the Hamiltonian to reuse.")

    if args is None:
        args = {}

    use_mesolve = (
        (c_ops and len(c_ops) > 0)
        or (not rho0.isket)
        or (isinstance(H, Qobj) and H.issuper)
        or (isinstance(H, QobjEvo) and H.issuper)
        or (isinstance(H, list) and isinstance(H[0], Qobj) and H[0].issuper)
        or (not isinstance(H, (Qobj, QobjEvo))
            and callable(H)
            and H(0., args).issuper)
        or (not isinstance(H, (Qobj, QobjEvo))
            and callable(H))
    )

    if not use_mesolve:
        return sesolve(H, rho0, tlist, e_ops=e_ops, args=args, options=options,
                       progress_bar=progress_bar, _safe_mode=_safe_mode)

    if isket(rho0):
        rho0 = ket2dm(rho0)
    if (not (rho0.isoper or rho0.issuper)) or (rho0.dims[0] != rho0.dims[1]):
        raise ValueError(
            "input state must be a pure state vector, square density matrix, "
            "or superoperator"
        )

    if isinstance(H, SolverSystem):
        ss = H
    else:
        H = QobjEvo(H, args=args, tlist=tlist)
        ss = _mesolve_QobjEvo(H, c_ops, tlist, args, options)
    """
    elif isinstance(H, (list, Qobj, QobjEvo)):
        ss = _mesolve_QobjEvo(H, c_ops, tlist, args, options)
    elif callable(H):
        ss = _mesolve_func_td(H, c_ops, rho0, tlist, args, options)
    else:
        raise Exception("Invalid H type")
    """

    func, ode_args = ss.makefunc(ss, rho0, args, e_ops, options)

    if _safe_mode:
        # This is to test safety of the function before starting the loop.
        v = rho0.full().ravel('F')
        func(0., v, *ode_args)[:, 0] + v

    res = _generic_ode_solve(func, ode_args, rho0, tlist, e_ops, options,
                             progress_bar, dims=rho0.dims)
    res.num_collapse = len(c_ops)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
# _mesolve_QobjEvo(H, c_ops, tlist, args, options)
def _mesolve_QobjEvo(H, c_ops, tlist, args, opt):
    """
    Prepare the system for the solver, H can be an QobjEvo.
    """
    H_td = QobjEvo(H, args, tlist=tlist)
    if not H_td.issuper:
        L_td = liouvillian(H_td)
    else:
        L_td = H_td
    for op in c_ops:
        op_td = QobjEvo(op, args, tlist=tlist)
        if not op_td.issuper:
            op_td = lindblad_dissipator(op_td)
        L_td += op_td

    # if opt.rhs_with_state:
    #    L_td._check_old_with_state()

    ss = SolverSystem()
    ss.H = L_td
    ss.makefunc = _qobjevo_set
    solver_safe["mesolve"] = ss
    return ss


def _test_liouvillian_dimensions(L_dims, rho_dims):
    """
    Raise ValueError if the dimensions of the Liouvillian and the density
    matrix or superoperator state are incompatible with the master equation.
    """
    if L_dims[0] != L_dims[1]:
        raise ValueError("Liouvillian had nonsquare dims: " + str(L_dims))
    if not ((L_dims[1] == rho_dims) or (L_dims[1] == rho_dims[0])):
        raise ValueError("".join([
            "incompatible Liouvillian and state dimensions: ",
            str(L_dims), " and ", str(rho_dims),
        ]))


def _wrap_matmul(t, state, cqobj, unstack):
    data = _data.dense.fast_from_numpy(state)
    if unstack:
        data = _data.column_unstack_dense(data, cqobj.shape[1], inplace=True)
    out = cqobj.matmul_data(t, data)
    if unstack:
        out = _data.column_stack_dense(out, inplace=True)
    return out.as_ndarray()


def _qobjevo_set(HS, rho0, args, e_ops, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args)
    if not (rho0.issuper or rho0.isoper or rho0.isket):
        raise TypeError("The unitary solver requires rho0 to be"
                        " a ket or dm as initial state"
                        " or a super operator as initial state.")
    _test_liouvillian_dimensions(H_td.dims, rho0.dims)
    return _wrap_matmul, (H_td, rho0.issuper)


# -----------------------------------------------------------------------------
# Master equation solver for python-function time-dependence.
#
class _LiouvillianFromFunc:
    def __init__(self, func, c_ops, rho_dims):
        self.f = func
        self.c_ops = c_ops
        self.rho_dims = rho_dims

    def H2L(self, t, rho, args):
        Ht = self.f(t, args)
        Lt = -1.0j * (spre(Ht) - spost(Ht))
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        for op in self.c_ops:
            Lt += op(t)
        return Lt.data

    def H2L_with_state(self, t, rho, args):
        Ht = self.f(t, rho, args)
        Lt = -1.0j * (spre(Ht) - spost(Ht))
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        for op in self.c_ops:
            Lt += op(t)
        return Lt.data

    def L(self, t, rho, args):
        Lt = self.f(t, args)
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        for op in self.c_ops:
            Lt += op(t)
        return Lt.data

    def L_with_state(self, t, rho, args):
        Lt = self.f(t, rho, args)
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        for op in self.c_ops:
            Lt += op(t)
        return Lt.data


def _mesolve_func_td(L_func, c_op_list, rho0, tlist, args, opt):
    """
    Evolve the density matrix using an ODE solver with time dependent
    Hamiltonian.
    """
    c_ops = []
    for op in c_op_list:
        op_td = QobjEvo(op, args, tlist=tlist, copy=False)
        if not op_td.issuper:
            c_ops += [lindblad_dissipator(op_td)]
        else:
            c_ops += [op_td]
    if c_op_list:
        c_ops_ = [sum(c_ops)]
    else:
        c_ops_ = []

    if False:  # TODO check: old version was `opt.rhs_with_state`
        state0 = rho0.full().ravel("F")
        obj = L_func(0., state0, args)
        if not issuper(obj):
            L_func = _LiouvillianFromFunc(L_func, c_ops_).H2L_with_state
        else:
            L_func = _LiouvillianFromFunc(L_func, c_ops_).L_with_state
    else:
        obj = L_func(0., args)
        L_func = L_api.L if issuper(obj) else L_api.H2L
    ss = SolverSystem()
    ss.L = L_func
    ss.makefunc = _Lfunc_set
    solver_safe["mesolve"] = ss
    return ss


def _Lfunc_set(HS, rho0, args, e_ops, opt):
    """
    From the system, get the ode function and args
    """
    L_func = HS.L
    if issuper(rho0):
        func = _ode_super_func_td
    else:
        func = _ode_rho_func_td

    return func, (L_func, args)


def _ode_rho_func_td(t, y, L_func, args):
    L = L_func(t, y, args)
    data = _data.dense.fast_from_numpy(y)
    return _data.matmul(L, data, dtype=_data.Dense).as_ndarray()


def _ode_super_func_td(t, y, L_func, args):
    L = L_func(t, y, args)
    data = _data.column_unstack_dense(_data.dense.fast_from_numpy(y),
                                      L.shape[1],
                                      inplace=True)
    matmul = _data.matmul(L, data, dtype=_data.Dense)
    return _data.column_stack_dense(matmul, inplace=True).as_ndarray()

# -----------------------------------------------------------------------------
# Generic ODE solver: shared code among the various ODE solver
# -----------------------------------------------------------------------------

def _generic_ode_solve(func, ode_args, rho0, tlist, e_ops, opt,
                       progress_bar, dims=None):
    """
    Internal function for solving ME.
    Calculate the required expectation values or invoke
    callback function at each time step.
    """
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This function is made similar to sesolve's one for futur merging in a
    # solver class
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # prepare output array
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "mesolve"
    output.times = tlist
    size = rho0.shape[0]

    initial_vector = rho0.full().ravel('F')

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
            # fall back on storing states
            opt['store_states'] = True
        else:
            for op in e_ops:
                if not isinstance(op, Qobj) and callable(op):
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
                    continue
                if op.dims != rho0.dims:
                    raise TypeError(f"e_ops dims ({op.dims}) are not "
                                    f"compatible with the state's "
                                    f"({rho0.dims})")
                e_ops_data.append(spre(op).data)
                dtype = (np.float64 if op.isherm and rho0.isherm
                         else np.complex128)
                output.expect.append(np.zeros(n_tsteps, dtype=dtype))
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt['store_states']:
        output.states = []

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

        if opt['store_states'] or expt_callback or n_expt_op:
            cdata = get_curr_state_data(r)

        if opt['store_states']:
            # Unstacking with a copying operation keeps us safe if a later call
            # unstacks the columns again.
            if cdata.shape[0] != size:
                cdata = _data.column_unstack_dense(cdata, size, inplace=False)
            output.states.append(Qobj(cdata,
                                      dims=dims, type=rho0.type, copy=False))

        if expt_callback:
            # use callback method
            if cdata.shape[0] != size:
                cdata = _data.column_unstack_dense(cdata, size, inplace=False)
            output.expect.append(e_ops(t, Qobj(cdata,
                                               dims=dims, type=rho0.type,
                                               copy=False)))

        for m in range(n_expt_op):
            if not isinstance(e_ops[m], Qobj) and callable(e_ops[m]):
                if cdata.shape[0] != size:
                    qdata = _data.column_unstack_dense(cdata, size, inplace=False)
                else:
                    qdata = cdata
                output.expect[m][t_idx] = e_ops[m](t, Qobj(qdata, dims=dims))
            else:
                if cdata.shape[1] != 1:
                    qdata = _data.column_stack_dense(cdata, inplace=False)
                else:
                    qdata = cdata
                val = _data.expect_super(e_ops_data[m], qdata)
                if e_ops[m].isherm and rho0.isherm:
                    val = val.real
                output.expect[m][t_idx] = val

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if opt['store_final_state']:
        cdata = get_curr_state_data(r)
        matrix = _data.column_unstack_dense(cdata, size)
        output.final_state = Qobj(matrix,
                                  dims=dims, type=rho0.type,
                                  isherm=rho0.isherm or None, copy=False)

    return output

"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['mesolve']

import numpy as np
import scipy.integrate
from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.superoperator import spre, spost, liouvillian, vec2mat, lindblad_dissipator
from qutip.expect import expect_rho_vec
from qutip.solver import Options, Result, solver_safe, SolverSystem
from qutip.cy.spmatfuncs import spmv
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.states import ket2dm
from qutip.sesolve import sesolve
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.qobjevo import QobjEvo

from qutip.cy.openmp.utilities import check_use_openmp

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
    should be an instance of :class:`qutip.solver.Options`. Many ODE
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

    options : None / :class:`qutip.solver.Options`
        with options for the solver.

    progress_bar : None / BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
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
    if issuper(rho0) and not e_ops == []:
        raise TypeError("Must have e_ops = [] when initial condition rho0 is" +
                " a superoperator.")

    if options is None:
        options = Options()
    if options.rhs_reuse and not isinstance(H, SolverSystem):
        # TODO: deprecate when going to class based solver.
        if "mesolve" in solver_safe:
            # print(" ")
            H = solver_safe["mesolve"]
        else:
            pass
            # raise Exception("Could not find the Hamiltonian to reuse.")

    if args is None:
        args = {}

    check_use_openmp(options)

    use_mesolve = ((c_ops and len(c_ops) > 0)
                   or (not isket(rho0))
                   or (isinstance(H, Qobj) and issuper(H))
                   or (isinstance(H, QobjEvo) and issuper(H.cte))
                   or (isinstance(H, list) and isinstance(H[0], Qobj) and
                            issuper(H[0]))
                   or (not isinstance(H, (Qobj, QobjEvo)) and callable(H) and
                            not options.rhs_with_state and issuper(H(0., args)))
                   or (not isinstance(H, (Qobj, QobjEvo)) and callable(H) and
                            options.rhs_with_state))

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
    elif isinstance(H, (list, Qobj, QobjEvo)):
        ss = _mesolve_QobjEvo(H, c_ops, tlist, args, options)
    elif callable(H):
        ss = _mesolve_func_td(H, c_ops, rho0, tlist, args, options)
    else:
        raise Exception("Invalid H type")

    func, ode_args = ss.makefunc(ss, rho0, args, e_ops, options)

    if _safe_mode:
        # This is to test safety of the function before starting the loop.
        v = rho0.full().ravel('F')
        func(0., v, *ode_args) + v

    res = _generic_ode_solve(func, ode_args, rho0, tlist, e_ops, options,
                             progress_bar, dims=rho0.dims)
    res.num_collapse = len(c_ops)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#_mesolve_QobjEvo(H, c_ops, tlist, args, options)
def _mesolve_QobjEvo(H, c_ops, tlist, args, opt):
    """
    Prepare the system for the solver, H can be an QobjEvo.
    """
    H_td = QobjEvo(H, args, tlist=tlist)
    if not issuper(H_td.cte):
        L_td = liouvillian(H_td)
    else:
        L_td = H_td
    for op in c_ops:
        # We want to avoid passing tlist where it isn't necessary, to allow a
        # Hamiltonian/Liouvillian which already _has_ time-dependence not equal
        # to the mesolve evaluation times to be used in conjunction with
        # time-independent c_ops.  If we _always_ pass it, it may appear to
        # QobjEvo that there is a tlist mismatch, even though it is not used.
        if isinstance(op, Qobj):
            op_td = QobjEvo(op)
        elif isinstance(op, QobjEvo):
            op_td = QobjEvo(op, args)
        else:
            op_td = QobjEvo(op, args, tlist=tlist)
        if not issuper(op_td.cte):
            op_td = lindblad_dissipator(op_td)
        L_td += op_td

    if opt.rhs_with_state:
        L_td._check_old_with_state()

    nthread = opt.openmp_threads if opt.use_openmp else 0
    L_td.compile(omp=nthread)

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


def _qobjevo_set(HS, rho0, args, e_ops, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.solver_set_args(args, rho0, e_ops)
    if issuper(rho0):
        func = H_td.compiled_qobjevo.ode_mul_mat_f_vec
    elif rho0.isket or rho0.isoper:
        func = H_td.compiled_qobjevo.mul_vec
    else:
        # Should be caught earlier in mesolve.
        raise ValueError("rho0 must be a ket, density matrix or superoperator")
    _test_liouvillian_dimensions(H_td.cte.dims, rho0.dims)
    return func, ()


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
        Lt = Lt.data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt

    def H2L_with_state(self, t, rho, args):
        Ht = self.f(t, rho, args)
        Lt = -1.0j * (spre(Ht) - spost(Ht))
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        Lt = Lt.data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt

    def L(self, t, rho, args):
        Lt = self.f(t, args)
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        Lt = Lt.data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt

    def L_with_state(self, t, rho, args):
        Lt = self.f(t, rho, args)
        _test_liouvillian_dimensions(Lt.dims, self.rho_dims)
        Lt = Lt.data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt


def _mesolve_func_td(L_func, c_op_list, rho0, tlist, args, opt):
    """
    Evolve the density matrix using an ODE solver with time dependent
    Hamiltonian.
    """
    c_ops = []
    for op in c_op_list:
        td = QobjEvo(op, args, tlist=tlist, copy=False)
        c_ops.append(td if td.cte.issuper else lindblad_dissipator(td))
    c_ops_ = [sum(c_ops)] if c_op_list else []
    L_api = _LiouvillianFromFunc(L_func, c_ops_, rho0.dims)
    if opt.rhs_with_state:
        obj = L_func(0., rho0.full().ravel("F"), args)
        L_func = L_api.L_with_state if issuper(obj) else L_api.H2L_with_state
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
    return spmv(L, y)

def _ode_super_func_td(t, y, L_func, args):
    L = L_func(t, y, args)
    ym = vec2mat(y)
    return (L*ym).ravel('F')

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
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    if ode_args:
        r.set_f_params(*ode_args)
    r.set_initial_value(initial_vector, tlist[0])

    e_ops_data = []
    output.expect = []
    need_qobj_state = opt.store_states
    if callable(e_ops):
        n_expt_op = 0
        expt_callback = True
        output.num_expect = 1
        need_qobj_state = True
    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False
        output.num_expect = n_expt_op
        if n_expt_op == 0:
            # fall back on storing states
            opt.store_states = True
            need_qobj_state = True
        else:
            for op in e_ops:
                if not isinstance(op, Qobj) and callable(op):
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
                    need_qobj_state = True
                    e_ops_data.append(None)
                    continue
                if op.dims != rho0.dims:
                    raise TypeError(f"e_ops dims ({op.dims}) are not "
                                    f"compatible with the state's "
                                    f"({rho0.dims})")
                e_ops_data.append(spre(op).data)
                if op.isherm and rho0.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt.store_states:
        output.states = []

    def get_curr_state_data(r):
        return vec2mat(r.y)

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

        if need_qobj_state:
            cdata = get_curr_state_data(r)
            fdata = dense2D_to_fastcsr_fmode(cdata, size, size)

            # Try to guess if there is a fast path for rho_t
            if issuper(rho0) or not rho0.isherm:
                rho_t = Qobj(fdata, dims=dims)
            else:
                rho_t = Qobj(fdata, dims=dims, fast="mc-dm")

        if opt.store_states:
            output.states.append(rho_t)

        if expt_callback:
            # use callback method
            output.expect.append(e_ops(t, rho_t))

        for m in range(n_expt_op):
            if not isinstance(e_ops[m], Qobj) and callable(e_ops[m]):
                output.expect[m][t_idx] = e_ops[m](t, rho_t)
            else:
                output.expect[m][t_idx] = expect_rho_vec(e_ops_data[m], r.y,
                                                         e_ops[m].isherm
                                                         and rho0.isherm)

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if opt.store_final_state:
        cdata = get_curr_state_data(r)
        output.final_state = Qobj(cdata, dims=dims,
                                  isherm=rho0.isherm or None)

    return output

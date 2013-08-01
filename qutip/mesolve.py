# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###############################################################################
"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

import os
import types
from functools import partial
import numpy as np
import scipy.sparse as sp
import scipy.integrate
from scipy.linalg import norm
import warnings

from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.superoperator import spre, spost, liouvillian_fast, mat2vec, vec2mat
from qutip.expect import expect, expect_rho_vec
from qutip.odeoptions import Odeoptions
from qutip.cyQ.spmatfuncs import cy_ode_rhs, cy_ode_rho_func_td
from qutip.cyQ.codegen import Codegen
from qutip.rhs_generate import rhs_generate
from qutip.odedata import Odedata
from qutip.states import ket2dm
from qutip.odechecks import _ode_checks
from qutip.odeconfig import odeconfig
from qutip.settings import debug

from qutip.sesolve import (_sesolve_list_func_td, _sesolve_list_str_td,
                           _sesolve_list_td, _sesolve_func_td, _sesolve_const)

from qutip.gui.progressbar import BaseProgressBar

if debug:
    import inspect


# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def mesolve(H, rho0, tlist, c_ops, e_ops, args={}, options=None,
            progress_bar=BaseProgressBar()):
    """
    Master equation evolution of a density matrix for a given Hamiltonian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian (`H`) and an [optional] set of collapse operators
    (`c_op_list`), by integrating the set of ordinary differential equations
    that define the system. In the absense of collase operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    **Time-dependent operators**

    For problems with time-dependent problems `H` and `c_ops` can be callback
    functions that takes two arguments, time and `args`, and returns the
    Hamiltonian or Liuovillian for the system at that point in time
    (*callback format*).

    Alternatively, `H` and `c_ops` can be a specified in a nested-list format
    where each element in the list is a list of length 2, containing an
    operator (:class:`qutip.qobj`) at the first element and where the
    second element is either a string (*list string format*) or a callback
    function (*list callback format*) that evaluates to the time-dependent
    coefficient for the corresponding operator.

    *Examples*

        H = [[H0, 'sin(w*t)'], [H1, 'sin(2*w*t)']]

        H = [[H0, f0_t], [H1, f1_t]]

        where f0_t and f1_t are python functions with signature f_t(t, args).

    In the *list string format* and *list callback format*, the string
    expression and the callback function must evaluate to a real or complex
    number (coefficient for the corresponding operator).

    In all cases of time-dependent operators, `args` is a dictionary of
    parameters that is used when evaluating operators. It is passed to the
    callback functions as second argument

    .. note::

        If an element in the list-specification of the Hamiltonian or
        the list of collapse operators are in super-operator for it will be
        added to the total Liouvillian of the problem with out further
        transformation. This allows for using mesolve for solving master
        equations that are not on standard Lindblad form.

    .. note::

        On using callback function: mesolve transforms all :class:`qutip.qobj`
        objects to sparse matrices before handing the problem to the integrator
        function. In order for your callback function to work correctly, pass
        all :class:`qutip.qobj` objects that are used in constructing the
        Hamiltonian via args. odesolve will check for :class:`qutip.qobj` in
        `args` and handle the conversion to sparse matrices. All other
        :class:`qutip.qobj` objects that are not passed via `args` will be
        passed on to the integrator to scipy who will raise an NotImplemented
        exception.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, or a callback function for time-dependent
        Hamiltonians.

    rho0 : :class:`qutip.qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of :class:`qutip.qobj`
        single collapse operator, or list of collapse operators.

    e_ops : list of :class:`qutip.qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.Odeoptions`
        with options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.odedata`

        An instance of the class :class:`qutip.odedata`, which contains either
        an *array* of expectation values for the times specified by `tlist`, or
        an *array* or state vectors or density matrices corresponding to the
        times in `tlist` [if `e_ops` is an empty list], or
        nothing if a callback function was given inplace of operators for
        which to calculate the expectation values.

    """

    # check whether c_ops or e_ops is is a single operator
    # if so convert it to a list containing only that operator
    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    # check for type (if any) of time-dependent inputs
    n_const, n_func, n_str = _ode_checks(H, c_ops)

    if options is None:
        options = Odeoptions()

    if (not options.rhs_reuse) or (not odeconfig.tdfunc):
        # reset odeconfig collapse and time-dependence flags to default values
        odeconfig.reset()

    res = None
    
    #
    # dispatch the appropriate solver
    #
    if ((c_ops and len(c_ops) > 0)
        or (not isket(rho0))
        or (isinstance(H, Qobj) and issuper(H))
        or (isinstance(H, list) and
            isinstance(H[0], Qobj) and issuper(H[0]))):

        #
        # we have collapse operators
        #

        #
        # find out if we are dealing with all-constant hamiltonian and
        # collapse operators or if we have at least one time-dependent
        # operator. Then delegate to appropriate solver...
        #

        if isinstance(H, Qobj):
            # constant hamiltonian
            if n_func == 0 and n_str == 0:
                # constant collapse operators
                res = _mesolve_const(H, rho0, tlist, c_ops,
                                     e_ops, args, options,
                                     progress_bar)
            elif n_str > 0:
                # constant hamiltonian but time-dependent collapse
                # operators in list string format
                res = _mesolve_list_str_td([H], rho0, tlist, c_ops,
                                           e_ops, args, options,
                                           progress_bar)
            elif n_func > 0:
                # constant hamiltonian but time-dependent collapse
                # operators in list function format
                res = _mesolve_list_func_td([H], rho0, tlist, c_ops,
                                            e_ops, args, options,
                                            progress_bar)

        elif isinstance(H, (types.FunctionType,
                            types.BuiltinFunctionType, partial)):
            # old style time-dependence: must have constant collapse operators
            if n_str > 0:  # or n_func > 0:
                raise TypeError("Incorrect format: function-format " +
                                "Hamiltonian cannot be mixed with " +
                                "time-dependent collapse operators.")
            else:
                res = _mesolve_func_td(H, rho0, tlist, c_ops,
                                       e_ops, args, options,
                                       progress_bar)

        elif isinstance(H, list):
            # determine if we are dealing with list of [Qobj, string] or
            # [Qobj, function] style time-dependences (for pure python and
            # cython, respectively)
            if n_func > 0:
                res = _mesolve_list_func_td(H, rho0, tlist, c_ops,
                                            e_ops, args, options,
                                            progress_bar)
            else:
                res = _mesolve_list_str_td(H, rho0, tlist, c_ops,
                                           e_ops, args, options,
                                           progress_bar)

        else:
            raise TypeError("Incorrect specification of Hamiltonian " +
                            "or collapse operators.")

    else:
        #
        # no collapse operators: unitary dynamics
        #
        if n_func > 0:
            res = _sesolve_list_func_td(H, rho0, tlist,
                                         e_ops, args, options, progress_bar)
        elif n_str > 0:
            res = _sesolve_list_str_td(H, rho0, tlist,
                                        e_ops, args, options, progress_bar)
        elif isinstance(H, (types.FunctionType,
                            types.BuiltinFunctionType, partial)):
            res = _sesolve_func_td(H, rho0, tlist,
                                   e_ops, args, options, progress_bar)
        else:
            res = _sesolve_const(H, rho0, tlist,
                                 e_ops, args, options, progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n] for n, e in enumerate(e_ops_dict.keys())}

    return res

# -----------------------------------------------------------------------------
# A time-dependent disipative master equation on the list-function format
#
def _mesolve_list_func_td(H_list, rho0, tlist, c_list, e_ops, args, opt,
                          progress_bar):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if isket(rho0):
        rho0 = rho0 * rho0.dag()

    #
    # construct liouvillian in list-function format
    #
    L_list = []
    if opt.rhs_with_state:
        constant_func = lambda x, y, z: 1.0
    else:
        constant_func = lambda x, y: 1.0

    # add all hamitonian terms to the lagrangian list
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec
            h_coeff = constant_func

        elif isinstance(h_spec, list) and isinstance(h_spec[0], Qobj):
            h = h_spec[0]
            h_coeff = h_spec[1]

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected callback function)")

        if isoper(h):
            L_list.append([(-1j * (spre(h) - spost(h))).data, h_coeff, False])

        elif issuper(h):
            L_list.append([h.data, h_coeff, False])

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected operator or superoperator)")

    # add all collapse operators to the lagrangian list
    for c_spec in c_list:

        if isinstance(c_spec, Qobj):
            c = c_spec
            c_coeff = constant_func
            c_square = False

        elif isinstance(c_spec, list) and isinstance(c_spec[0], Qobj):
            c = c_spec[0]
            c_coeff = c_spec[1]
            c_square = True

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected callback function)")

        if isoper(c):
            # cdc = c.dag() * c
            # L_list.append([(spre(c) * spost(c.dag()) - 0.5 * spre(cdc)
            #                                         - 0.5 * spost(cdc)).data,
            #               c_coeff, c_square])

            cdc = c.dag() * c
            L_list.append([liouvillian_fast(None, [c], data_only=True),
                           c_coeff, c_square])

        elif issuper(c):
            L_list.append([c.data, c_coeff, c_square])

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected operator or " +
                            "superoperator)")

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    if opt.rhs_with_state:
        r = scipy.integrate.ode(drho_list_td_with_state)
    else:
        r = scipy.integrate.ode(drho_list_td)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(L_list, args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, rho0, tlist, e_ops, opt, progress_bar)


#
# evaluate drho(t)/dt according to the master equation using the
# [Qobj, function] style time dependence API
#
def drho_list_td(t, rho, L_list, args):

    L = L_list[0][0] * L_list[0][1](t, args)
    for n in range(1, len(L_list)):
        #
        # L_args[n][0] = the sparse data for a Qobj in super-operator form
        # L_args[n][1] = function callback giving the coefficient
        #
        if L_list[n][2]:
            L = L + L_list[n][0] * (L_list[n][1](t, args)) ** 2
        else:
            L = L + L_list[n][0] * L_list[n][1](t, args)

    return L * rho


def drho_list_td_with_state(t, rho, L_list, args):

    L = L_list[0][0] * L_list[0][1](t, rho, args)
    for n in range(1, len(L_list)):
        #
        # L_args[n][0] = the sparse data for a Qobj in super-operator form
        # L_args[n][1] = function callback giving the coefficient
        #
        if L_list[n][2]:
            L = L + L_list[n][0] * (L_list[n][1](t, rho, args)) ** 2
        else:
            L = L + L_list[n][0] * L_list[n][1](t, rho, args)

    return L * rho


# -----------------------------------------------------------------------------
# A time-dependent disipative master equation on the list-string format for
# cython compilation
#
def _mesolve_list_str_td(H_list, rho0, tlist, c_list, e_ops, args, opt,
                         progress_bar):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state: must be a density matrix
    #
    if isket(rho0):
        rho0 = rho0 * rho0.dag()

    #
    # construct liouvillian
    #
    Lconst = 0

    Ldata = []
    Linds = []
    Lptrs = []
    Lcoeff = []

    # loop over all hamiltonian terms, convert to superoperator form and
    # add the data of sparse matrix represenation to
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec

            if isoper(h):
                Lconst += -1j * (spre(h) - spost(h))
            elif issuper(h):
                Lconst += h
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Hamiltonian (expected operator or " +
                                "superoperator)")

        elif isinstance(h_spec, list):
            h = h_spec[0]
            h_coeff = h_spec[1]

            if isoper(h):
                L = -1j * (spre(h) - spost(h))
            elif issuper(h):
                L = h
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Hamiltonian (expected operator or " +
                                "superoperator)")

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append(h_coeff)

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected string format)")

    # loop over all collapse operators
    for c_spec in c_list:

        if isinstance(c_spec, Qobj):
            c = c_spec

            if isoper(c):
                cdc = c.dag() * c
                Lconst += spre(c) * spost(c.dag()) - 0.5 * spre(cdc) \
                                                   - 0.5 * spost(cdc)
            elif issuper(c):
                Lconst += c
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Liouvillian (expected operator or " +
                                "superoperator)")

        elif isinstance(c_spec, list):
            c = c_spec[0]
            c_coeff = c_spec[1]

            if isoper(c):
                cdc = c.dag() * c
                L = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) \
                                             - 0.5 * spost(cdc)
                c_coeff = "(" + c_coeff + ")**2"
            elif issuper(c):
                L = c
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Liouvillian (expected operator or " +
                                "superoperator)")

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append(c_coeff)

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected string format)")

    # add the constant part of the lagrangian
    if Lconst != 0:
        Ldata.append(Lconst.data.data)
        Linds.append(Lconst.data.indices)
        Lptrs.append(Lconst.data.indptr)
        Lcoeff.append("1.0")

    # the total number of liouvillian terms (hamiltonian terms +
    # collapse operators)
    n_L_terms = len(Ldata)

    #
    # setup ode args string: we expand the list Ldata, Linds and Lptrs into
    # and explicit list of parameters
    #
    string_list = []
    for k in range(n_L_terms):
        string_list.append("Ldata[%d], Linds[%d], Lptrs[%d]" % (k, k, k))
    for name, value in args.items():
        string_list.append(str(value))
    parameter_string = ",".join(string_list)

    #
    # generate and compile new cython code if necessary
    #
    if not opt.rhs_reuse or odeconfig.tdfunc is None:
        if opt.rhs_filename is None:
            odeconfig.tdname = "rhs" + str(odeconfig.cgen_num)
        else:
            odeconfig.tdname = opt.rhs_filename
        cgen = Codegen(h_terms=n_L_terms, h_tdterms=Lcoeff, args=args,
                       odeconfig=odeconfig)
        cgen.generate(odeconfig.tdname + ".pyx")

        code = compile('from ' + odeconfig.tdname + ' import cyq_td_ode_rhs',
                       '<string>', 'exec')
        exec(code, globals())
        odeconfig.tdfunc = cyq_td_ode_rhs

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(odeconfig.tdfunc)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    code = compile('r.set_f_params(' + parameter_string + ')',
                   '<string>', 'exec')
    exec(code)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, rho0, tlist, e_ops, opt, progress_bar)


# -----------------------------------------------------------------------------
# Master equation solver
#
def _mesolve_const(H, rho0, tlist, c_op_list, e_ops, args, opt,
                   progress_bar):
    """!
    Evolve the density matrix using an ODE solver, for constant hamiltonian
    and collapse operators.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if isket(rho0):
        # if initial state is a ket and no collapse operator where given,
        # fallback on the unitary schrodinger equation solver
        if len(c_op_list) == 0 and isoper(H):
            return _sesolve_const(H, rho0, tlist, e_ops, args, opt)

        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # construct liouvillian
    #
    if opt.tidy:
        H = H.tidyup(opt.atol)

    L = liouvillian_fast(H, c_op_list)

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(L.data.data, L.data.indices, L.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, rho0, tlist, e_ops, opt, progress_bar)


#
# evaluate drho(t)/dt according to the master eqaution
# [no longer used, replaced by cython function]
#
def _ode_rho_func(t, rho, L):
    return L * rho


# -----------------------------------------------------------------------------
# Master equation solver: deprecated in 2.0.0. No support for time-dependent
# collapse operators. Only used by the deprecated odesolve function.
#
def _mesolve_list_td(H_func, rho0, tlist, c_op_list, e_ops, args, opt,
                     progress_bar):
    """!
    Evolve the density matrix using an ODE solver with time dependent
    Hamiltonian.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if isket(rho0):
        # if initial state is a ket and no collapse operator where given,
        # fallback on the unitary schrodinger equation solver
        if len(c_op_list) == 0:
            return _sesolve_list_td(H_func, rho0, tlist, e_ops, args, opt)

        # Got a wave function as initial state: convert to density matrix.
        rho0 = ket2dm(rho0)

    #
    # construct liouvillian
    #
    if len(H_func) != 2:
        raise TypeError('Time-dependent Hamiltonian list must have two terms.')
    if not isinstance(H_func[0], (list, np.ndarray)) or len(H_func[0]) <= 1:
        raise TypeError('Time-dependent Hamiltonians must be a list ' +
                        'with two or more terms')
    if (not isinstance(H_func[1], (list, np.ndarray))) or \
       (len(H_func[1]) != (len(H_func[0]) - 1)):
        raise TypeError('Time-dependent coefficients must be list with ' +
                        'length N-1 where N is the number of ' +
                        'Hamiltonian terms.')

    if opt.rhs_reuse and odeconfig.tdfunc is None:
        rhs_generate(H_func, args)

    lenh = len(H_func[0])
    if opt.tidy:
        H_func[0] = [(H_func[0][k]).tidyup() for k in range(lenh)]
    L_func = [[liouvillian_fast(H_func[0][0], c_op_list)], H_func[1]]
    for m in range(1, lenh):
        L_func[0].append(liouvillian_fast(H_func[0][m], []))

    # create data arrays for time-dependent RHS function
    Ldata = [L_func[0][k].data.data for k in range(lenh)]
    Linds = [L_func[0][k].data.indices for k in range(lenh)]
    Lptrs = [L_func[0][k].data.indptr for k in range(lenh)]
    # setup ode args string
    string = ""
    for k in range(lenh):
        string += ("Ldata[%d], Linds[%d], Lptrs[%d]," % (k, k, k))

    if args:
        td_consts = args.items()
        for elem in td_consts:
            string += str(elem[1])
            if elem != td_consts[-1]:
                string += (",")

    # run code generator
    if not opt.rhs_reuse or odeconfig.tdfunc is None:
        if opt.rhs_filename is None:
            odeconfig.tdname = "rhs" + str(odeconfig.cgen_num)
        else:
            odeconfig.tdname = opt.rhs_filename
        cgen = Codegen(h_terms=n_L_terms, h_tdterms=Lcoeff, args=args,
                       odeconfig=odeconfig)
        cgen.generate(odeconfig.tdname + ".pyx")

        code = compile('from ' + odeconfig.tdname + ' import cyq_td_ode_rhs',
                       '<string>', 'exec')
        exec(code, globals())
        odeconfig.tdfunc = cyq_td_ode_rhs

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(odeconfig.tdfunc)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    code = compile('r.set_f_params(' + string + ')', '<string>', 'exec')
    exec(code)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, rho0, tlist, e_ops, opt, progress_bar)


# -----------------------------------------------------------------------------
# Master equation solver
#
def _mesolve_func_td(L_func, rho0, tlist, c_op_list, e_ops, args, opt,
                     progress_bar):
    """!
    Evolve the density matrix using an ODE solver with time dependent
    Hamiltonian.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if isket(rho0):
        rho0 = ket2dm(rho0)

    #
    # construct liouvillian
    #
    new_args = None

    if len(c_op_list) > 0:
        L_data = liouvillian_fast(None, c_op_list).data
    else:
        n, m = rho0.shape
        L_data = sp.csr_matrix((n ** 2, m ** 2), dtype=complex)

    if type(args) is dict:
        new_args = {}
        for key in args:
            if isinstance(args[key], Qobj):
                if isoper(args[key]):
                    new_args[key] = (
                        -1j * (spre(args[key]) - spost(args[key]))).data
                else:
                    new_args[key] = args[key].data
            else:
                new_args[key] = args[key]

    elif type(args) is list:
        new_args = []
        for arg in args:
            if isinstance(arg, Qobj):
                if isoper(arg):
                    new_args.append((-1j * (spre(arg) - spost(arg))).data)
                else:
                    new_args.append(arg.data)
            else:
                new_args.append(arg)

    else:
        if isinstance(args, Qobj):
            if isoper(args):
                new_args = (-1j * (spre(args) - spost(args))).data
            else:
                new_args = args.data
        else:
            new_args = args

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    if not opt.rhs_with_state:
        r = scipy.integrate.ode(cy_ode_rho_func_td)
    else:
        r = scipy.integrate.ode(_ode_rho_func_td_with_state)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(L_data, L_func, new_args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, rho0, tlist, e_ops, opt, progress_bar)


#
# evaluate drho(t)/dt according to the master eqaution
#
def _ode_rho_func_td(t, rho, L0, L_func, args):
    L = L0 + L_func(t, args)
    return L * rho


#
# evaluate drho(t)/dt according to the master eqaution
#
def _ode_rho_func_td_with_state(t, rho, L0, L_func, args):
    L = L0 + L_func(t, rho, args)
    return L * rho


# -----------------------------------------------------------------------------
# Generic ODE solver: shared code amoung the various ODE solver
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Solve an ODE which solver parameters already setup (r). Calculate the
# required expectation values or invoke callback function at each time step.
#
def _generic_ode_solve(r, rho0, tlist, e_ops, opt, progress_bar):
    """
    Internal function for solving ME.
    """

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]
    e_sops_data = []

    output = Odedata()
    output.solver = "mesolve"
    output.times = tlist

    if opt.store_states:
        output.states = []

    if isinstance(e_ops, types.FunctionType):
        n_expt_op = 0
        expt_callback = True

    elif isinstance(e_ops, list):

        n_expt_op = len(e_ops)
        expt_callback = False

        if n_expt_op == 0:
            # fallback on storing states
            output.states = []
            opt.store_states = True
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in e_ops:
                e_sops_data.append(spre(op).data)
                if op.isherm and rho0.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))

    else:
        raise TypeError("Expectation parameter must be a list or a function")

    #
    # start evolution
    #
    progress_bar.start(n_tsteps)

    rho = Qobj(rho0)

    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)

        if not r.successful():
            break

        if opt.store_states or expt_callback:
            rho.data = vec2mat(r.y)

            if opt.store_states:
                output.states.append(Qobj(rho))

            if expt_callback:
                # use callback method
                e_ops(t, rho)

        for m in range(n_expt_op):
            if output.expect[m].dtype == complex:
                output.expect[m][t_idx] = expect_rho_vec(e_sops_data[m], r.y)
            else:
                output.expect[m][t_idx] = np.real(
                    expect_rho_vec(e_sops_data[m], r.y))

        r.integrate(r.t + dt)

    progress_bar.finished()

    if not opt.rhs_reuse and odeconfig.tdname is not None:
        try:
            os.remove(odeconfig.tdname + ".pyx")
        except:
            pass

    if opt.store_final_state:
        rho.data = vec2mat(r.y)
        output.final_state = Qobj(rho)

    return output


# -----------------------------------------------------------------------------
# Old style API below.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def odesolve(H, rho0, tlist, c_op_list, e_ops, args=None, options=None):
    """
    Master equation evolution of a density matrix for a given Hamiltonian.

    Evolution of a state vector or density matrix (`rho0`) for a given
    Hamiltonian (`H`) and set of collapse operators (`c_op_list`), by
    integrating the set of ordinary differential equations that define the
    system. The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`).

    For problems with time-dependent Hamiltonians, `H` can be a callback
    function that takes two arguments, time and `args`, and returns the
    Hamiltonian at that point in time. `args` is a list of parameters that is
    passed to the callback function `H` (only used for time-dependent
    Hamiltonians).

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, or a callback function for time-dependent
        Hamiltonians.

    rho0 : :class:`qutip.qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_op_list : list of :class:`qutip.qobj`
        list of collapse operators.

    e_ops : list of :class:`qutip.qobj` / callback function
        list of operators for which to evaluate expectation values.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.Odeoptions`
        with options for the ODE solver.


    Returns
    -------
    output :array
    Expectation values of wavefunctions/density matrices
    for the times specified by `tlist`.

    Notes
    -----
    On using callback function: odesolve transforms all :class:`qutip.qobj`
    objects to sparse matrices before handing the problem to the integrator
    function. In order for your callback function to work correctly, pass
    all :class:`qutip.qobj` objects that are used in constructing the
    Hamiltonian via args. odesolve will check for :class:`qutip.qobj` in
    `args` and handle the conversion to sparse matrices. All other
    :class:`qutip.qobj` objects that are not passed via `args` will be
    passed on to the integrator to scipy who will raise an NotImplemented
    exception.

    Deprecated in QuTiP 2.0.0. Use :func:`mesolve` instead.

    """

    warnings.warn("odesolve is deprecated since 2.0.0. Use mesolve instead.",
                  DeprecationWarning)

    if debug:
        print(inspect.stack()[0][3])

    if options is None:
        options = Odeoptions()

    if (c_op_list and len(c_op_list) > 0) or not isket(rho0):
        if isinstance(H, list):
            output = _mesolve_list_td(H, rho0, tlist,
                                      c_op_list, e_ops, args, options,
                                      BaseProgressBar())
        if isinstance(H, (types.FunctionType,
                          types.BuiltinFunctionType, partial)):
            output = _mesolve_func_td(H, rho0, tlist,
                                      c_op_list, e_ops, args, options,
                                      BaseProgressBar())
        else:
            output = _mesolve_const(H, rho0, tlist,
                                    c_op_list, e_ops, args, options,
                                    BaseProgressBar())
    else:
        if isinstance(H, list):
            output = _sesolve_list_td(H, rho0, tlist, e_ops, args, options,
                                      BaseProgressBar())
        if isinstance(H, (types.FunctionType,
                          types.BuiltinFunctionType, partial)):
            output = _sesolve_func_td(H, rho0, tlist, e_ops, args, options,
                                      BaseProgressBar())
        else:
            output = _sesolve_const(H, rho0, tlist, e_ops, args, options,
                                    BaseProgressBar())

    if len(e_ops) > 0:
        return output.expect
    else:
        return output.states

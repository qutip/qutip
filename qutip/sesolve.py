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
This module provides solvers for the unitary Schrodinger equation.
"""

import os
import types
from functools import partial
import numpy as np
import scipy.sparse as sp
import scipy.integrate
from scipy.linalg import norm

from qutip.qobj import Qobj, isket
from qutip.expect import expect
from qutip.rhs_generate import rhs_generate
from qutip.odedata import Odedata
from qutip.odeoptions import Odeoptions
from qutip.odeconfig import odeconfig
from qutip.odechecks import _ode_checks
from qutip.settings import debug
from qutip.cyQ.spmatfuncs import (cy_ode_rhs,
                                  cy_ode_psi_func_td,
                                  cy_ode_psi_func_td_with_state)
from qutip.cyQ.codegen import Codegen

from qutip.gui.progressbar import BaseProgressBar

if debug:
    import inspect


def sesolve(H, rho0, tlist, expt_ops, args={}, options=None,
            progress_bar=BaseProgressBar()):
    """
    Schrodinger equation evolution of a state vector for a given Hamiltonian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian (`H`), by integrating the set of ordinary differential
    equations that define the system.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`expt_ops`). If expt_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, or a callback function for time-dependent
        Hamiltonians.

    rho0 : :class:`qutip.qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    expt_ops : list of :class:`qutip.qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.odedata`

        An instance of the class :class:`qutip.odedata`, which contains either
        an *array* of expectation values for the times specified by `tlist`, or
        an *array* or state vectors or density matrices corresponding to the
        times in `tlist` [if `expt_ops` is an empty list], or
        nothing if a callback function was given inplace of operators for
        which to calculate the expectation values.

    """

    if isinstance(expt_ops, Qobj):
        expt_ops = [expt_ops]

    # check for type (if any) of time-dependent inputs
    n_const, n_func, n_str = _ode_checks(H, [])

    if options is None:
        options = Odeoptions()

    if (not options.rhs_reuse) or (not odeconfig.tdfunc):
        # reset odeconfig time-dependence flags to default values
        odeconfig.reset()

    if n_func > 0:
        return _sesolve_list_func_td(H, rho0, tlist, expt_ops, args, options,
                                     progress_bar)
    elif n_str > 0:
        return _sesolve_list_str_td(H, rho0, tlist, expt_ops, args, options,
                                    progress_bar)
    elif isinstance(H, (types.FunctionType, types.BuiltinFunctionType, partial)):
        return _sesolve_func_td(H, rho0, tlist, expt_ops, args, options,
                                progress_bar)
    else:
        return _sesolve_const(H, rho0, tlist, expt_ops, args, options,
                              progress_bar)


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#
def _sesolve_list_func_td(H_list, psi0, tlist, expt_ops, args, opt,
                          progress_bar):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if not isket(psi0):
        raise TypeError("The unitary solver requires a ket as initial state")

    #
    # construct liouvillian in list-function format
    #
    L_list = []
    if not opt.rhs_with_state:
        constant_func = lambda x, y: 1.0
    else:
        constant_func = lambda x, y, z: 1.0

    # add all hamitonian terms to the lagrangian list
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec
            h_coeff = constant_func

        elif isinstance(h_spec, list):
            h = h_spec[0]
            h_coeff = h_spec[1]

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected callback function)")

        L_list.append([-1j * h.data, h_coeff])

    L_list_and_args = [L_list, args]

    #
    # setup integrator
    #
    initial_vector = psi0.full().ravel()
    if not opt.rhs_with_state:
        r = scipy.integrate.ode(psi_list_td)
    else:
        r = scipy.integrate.ode(psi_list_td_with_state)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(L_list_and_args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, psi0, tlist, expt_ops, opt, progress_bar)


#
# evaluate dpsi(t)/dt according to the master equation using the
# [Qobj, function] style time dependence API
#
def psi_list_td(t, psi, H_list_and_args):

    H_list = H_list_and_args[0]
    args = H_list_and_args[1]

    H = H_list[0][0] * H_list[0][1](t, args)
    for n in range(1, len(H_list)):
        #
        # args[n][0] = the sparse data for a Qobj in operator form
        # args[n][1] = function callback giving the coefficient
        #
        H = H + H_list[n][0] * H_list[n][1](t, args)

    return H * psi


def psi_list_td_with_state(t, psi, H_list_and_args):

    H_list = H_list_and_args[0]
    args = H_list_and_args[1]

    H = H_list[0][0] * H_list[0][1](t, psi, args)
    for n in range(1, len(H_list)):
        #
        # args[n][0] = the sparse data for a Qobj in operator form
        # args[n][1] = function callback giving the coefficient
        #
        H = H + H_list[n][0] * H_list[n][1](t, psi, args)

    return H * psi


# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution) using
# a constant Hamiltonian.
#
def _sesolve_const(H, psi0, tlist, expt_ops, args, opt, progress_bar):
    """!
    Evolve the wave function using an ODE solver
    """
    if debug:
        print(inspect.stack()[0][3])

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")

    #
    # setup integrator.
    #
    initial_vector = psi0.full().ravel()
    r = scipy.integrate.ode(cy_ode_rhs)
    L = -1.0j * H
    r.set_f_params(L.data.data, L.data.indices, L.data.indptr)  # cython RHS
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)

    r.set_initial_value(initial_vector, tlist[0])

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, psi0, tlist, expt_ops, opt,
                              progress_bar, norm)


#
# evaluate dpsi(t)/dt [not used. using cython function is being used instead]
#
def _ode_psi_func(t, psi, H):
    return H * psi


# -----------------------------------------------------------------------------
# A time-dependent disipative master equation on the list-string format for
# cython compilation
#
def _sesolve_list_str_td(H_list, psi0, tlist, expt_ops, args, opt,
                         progress_bar):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state: must be a density matrix
    #
    if not isket(psi0):
        raise TypeError("The unitary solver requires a ket as initial state")

    #
    # construct liouvillian
    #
    Ldata = []
    Linds = []
    Lptrs = []
    Lcoeff = []

    # loop over all hamiltonian terms, convert to superoperator form and
    # add the data of sparse matrix representation to h_coeff
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec
            h_coeff = "1.0"

        elif isinstance(h_spec, list):
            h = h_spec[0]
            h_coeff = h_spec[1]

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected string format)")

        L = -1j * h

        Ldata.append(L.data.data)
        Linds.append(L.data.indices)
        Lptrs.append(L.data.indptr)
        Lcoeff.append(h_coeff)

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
    initial_vector = psi0.full().ravel()
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
    return _generic_ode_solve(r, psi0, tlist, expt_ops, opt, progress_bar)


# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians
#
def _sesolve_list_td(H_func, psi0, tlist, expt_ops, args, opt, progress_bar):
    """!
    Evolve the wave function using an ODE solver with time-dependent
    Hamiltonian.
    """

    if debug:
        print(inspect.stack()[0][3])

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")

    #
    # configure time-dependent terms and setup ODE solver
    #
    if len(H_func) != 2:
        raise TypeError('Time-dependent Hamiltonian list must have two terms.')
    if (not isinstance(H_func[0], (list, np.ndarray))) or \
       (len(H_func[0]) <= 1):
        raise TypeError('Time-dependent Hamiltonians must be a list with two '
                        + 'or more terms')
    if (not isinstance(H_func[1], (list, np.ndarray))) or \
       (len(H_func[1]) != (len(H_func[0]) - 1)):
        raise TypeError('Time-dependent coefficients must be list with ' +
                        'length N-1 where N is the number of ' +
                        'Hamiltonian terms.')
    tflag = 1
    if opt.rhs_reuse and odeconfig.tdfunc is None:
        print("No previous time-dependent RHS found.")
        print("Generating one for you...")
        rhs_generate(H_func, args)
    lenh = len(H_func[0])
    if opt.tidy:
        H_func[0] = [(H_func[0][k]).tidyup() for k in range(lenh)]
    # create data arrays for time-dependent RHS function
    Hdata = [-1.0j * H_func[0][k].data.data for k in range(lenh)]
    Hinds = [H_func[0][k].data.indices for k in range(lenh)]
    Hptrs = [H_func[0][k].data.indptr for k in range(lenh)]
    # setup ode args string
    string = ""
    for k in range(lenh):
        string += ("Hdata[" + str(k) + "],Hinds[" + str(k) +
                   "],Hptrs[" + str(k) + "],")

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
    initial_vector = psi0.full().ravel()
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
    return _generic_ode_solve(r, psi0, tlist, expt_ops, opt, progress_bar)


# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians
#
def _sesolve_func_td(H_func, psi0, tlist, expt_ops, args, opt, progress_bar):
    """!
    Evolve the wave function using an ODE solver with time-dependent
    Hamiltonian.
    """

    if debug:
        print(inspect.stack()[0][3])

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")

    #
    # setup integrator
    #
    new_args = None

    if type(args) is dict:
        new_args = {}
        for key in args:
            if isinstance(args[key], Qobj):
                new_args[key] = args[key].data
            else:
                new_args[key] = args[key]

    elif type(args) is list:
        new_args = []
        for arg in args:
            if isinstance(arg, Qobj):
                new_args.append(arg.data)
            else:
                new_args.append(arg)
    else:
        if isinstance(args, Qobj):
            new_args = args.data
        else:
            new_args = args

    initial_vector = psi0.full().ravel()

    if not opt.rhs_with_state:
        r = scipy.integrate.ode(cy_ode_psi_func_td)
    else:
        r = scipy.integrate.ode(cy_ode_psi_func_td_with_state)

    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(H_func, new_args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, psi0, tlist, expt_ops, opt, progress_bar)


#
# evaluate dpsi(t)/dt for time-dependent hamiltonian
#
def _ode_psi_func_td(t, psi, H_func, args):
    H = H_func(t, args)
    return -1j * (H * psi)


def _ode_psi_func_td_with_state(t, psi, H_func, args):
    H = H_func(t, psi, args)
    return -1j * (H * psi)


# -----------------------------------------------------------------------------
# Solve an ODE which solver parameters already setup (r). Calculate the
# required expectation values or invoke callback function at each time step.
#
def _generic_ode_solve(r, psi0, tlist, expt_ops, opt, progress_bar, 
                       state_norm_func=None):
    """
    Internal function for solving ODEs.
    """

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]

    output = Odedata()
    output.solver = "sesolve"
    output.times = tlist

    if opt.store_states:
        output.states = []

    if isinstance(expt_ops, types.FunctionType):
        n_expt_op = 0
        expt_callback = True

    elif isinstance(expt_ops, list):

        n_expt_op = len(expt_ops)
        expt_callback = False

        if n_expt_op == 0:
            # fallback on storing states
            output.states = []
            opt.store_states = True
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in expt_ops:
                if op.isherm and psi0.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    #
    # start evolution
    #
    progress_bar.start(n_tsteps)

    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)

        if not r.successful():
            break

        if state_norm_func:
            data = r.y / state_norm_func(r.y)
            r.set_initial_value(data, r.t)

        if opt.store_states:
            output.states.append(Qobj(r.y))

        if expt_callback:
            # use callback method
            expt_ops(t, Qobj(r.y))

        for m in range(n_expt_op):
            output.expect[m][t_idx] = expect(expt_ops[m], Qobj(r.y)) # optimize

        r.integrate(r.t + dt)

    progress_bar.finished()

    if not opt.rhs_reuse and odeconfig.tdname is not None:
        try:
            os.remove(odeconfig.tdname + ".pyx")
        except:
            pass

    if opt.store_final_state:
        output.final_state = Qobj(r.y)

    return output

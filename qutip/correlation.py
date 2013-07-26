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
###########################################################################

import numpy as np
import scipy.fftpack

from qutip.superoperator import *
from qutip.expect import expect
from qutip.mesolve import mesolve
from qutip.eseries import esval, esspec
from qutip.essolve import ode2es
from qutip.mcsolve import mcsolve
from qutip.steadystate import steadystate
from qutip.states import ket2dm
from qutip.odeoptions import Odeoptions
from qutip.settings import debug

if debug:
    import inspect


#------------------------------------------------------------------------------
# PUBLIC API
#------------------------------------------------------------------------------

def correlation_2op_1t(H, rho0, taulist, c_ops, a_op, b_op, solver="me",
                       reverse=False, args=None, options=Odeoptions()):
    """
    Calculate a two-operator two-time correlation function
    :math:`\left<A(\\tau)B(0)\\right>` or
    :math:`\left<A(0)B(\\tau)\\right>` (if `reverse=True`),
    using the quantum regression theorem and the evolution solver indicated by
    the *solver* parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If `rho0` is
        `None`, then the steady state will be used as initial state.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        operator A.

    b_op : :class:`qutip.qobj.Qobj`
        operator B.

    reverse : bool
        If `True`, calculate :math:`\left<A(0)B(\\tau)\\right>` instead of
        :math:`\left<A(\\tau)B(0)\\right>`.

    solver : str
        choice of solver (`me` for master-equation,
        `es` for exponential series and `mc` for Monte-carlo)

    Returns
    -------

    corr_vec: *array*
        An *array* of correlation values for the times specified by `taulist`

    """

    if debug:
        print(inspect.stack()[0][3])

    if solver == "me":
        return _correlation_me_2op_1t(H, rho0, taulist, c_ops, a_op, b_op,
                                      reverse, args=args, options=options)
    elif solver == "es":
        return _correlation_es_2op_1t(H, rho0, taulist, c_ops, a_op, b_op,
                                      reverse, args=args, options=options)
    elif solver == "mc":
        return _correlation_mc_2op_1t(H, rho0, taulist, c_ops, a_op, b_op,
                                      reverse, args=args, options=options)
    else:
        raise "Unrecognized choice of solver %s (use me, es or mc)." % solver


def correlation_2op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op, solver="me",
                       reverse=False, args=None, options=Odeoptions()):
    """
    Calculate a two-operator two-time correlation function on the form
    :math:`\left<A(t+\\tau)B(t)\\right>` or
    :math:`\left<A(t)B(t+\\tau)\\right>` (if `reverse=True`), using the
    quantum regression theorem and the evolution solver indicated by the
    *solver* parameter.


    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix :math:`\\rho(t_0)` (or state vector). If
        'rho0' is 'None', then the steady state will be used as initial state.

    tlist : *list* / *array*
        list of times for :math:`t`.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        operator A.

    b_op : :class:`qutip.qobj.Qobj`
        operator B.

    solver : str
        choice of solver (`me` for master-equation,
        `es` for exponential series and `mc` for Monte-carlo)

    reverse : bool
        If `True`, calculate :math:`\left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\left<A(t+\\tau)B(t)\\right>`.

    Returns
    -------

    corr_mat: *array*
        An 2-dimensional *array* (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional *array* of correlation values
        is returned instead.

    """

    if debug:
        print(inspect.stack()[0][3])

    if tlist is None:
        # only interested in correlation vs one time coordinate, so we can use
        # the ss solver with the supplied density matrix as initial state (in
        # place of the steady state)
        return correlation_2op_1t(H, rho0, taulist, c_ops, a_op, b_op, solver,
                                  reverse, args=args, options=options)

    if solver == "me":
        return _correlation_me_2op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, reverse, args=args, options=options)
    elif solver == "es":
        return _correlation_es_2op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, reverse, args=args, options=options)
    elif solver == "mc":
        return _correlation_mc_2op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, reverse, args=args, options=options)
    else:
        raise "Unrecognized choice of solver %s (use me, es or mc)." % solver


def correlation_4op_1t(H, rho0, taulist, c_ops, a_op, b_op, c_op, d_op,
                       solver="me", args=None, options=Odeoptions()):
    """
    Calculate the four-operator two-time correlation function on the from
    :math:`\left<A(0)B(\\tau)C(\\tau)D(0)\\right>` using the quantum regression
    theorem and the solver indicated by the 'solver' parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        operator A.

    b_op : :class:`qutip.qobj.Qobj`
        operator B.

    c_op : :class:`qutip.qobj.Qobj`
        operator C.

    d_op : :class:`qutip.qobj.Qobj`
        operator D.

    solver : str
        choice of solver (currently only `me` for master-equation)

    Returns
    -------

    corr_vec: *array*
        An *array* of correlation values for the times specified by `taulist`


    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.1.

    """

    if debug:
        print(inspect.stack()[0][3])

    if solver == "me":
        return _correlation_me_4op_1t(H, rho0, taulist, c_ops,
                                      a_op, b_op, c_op, d_op,
                                      args=args, options=options)
    else:
        raise NotImplementedError("Unrecognized choice of solver %s." % solver)


def correlation_4op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op, c_op, d_op,
                       solver="me", args=None, options=Odeoptions()):
    """
    Calculate the four-operator two-time correlation function on the from
    :math:`\left<A(t)B(t+\\tau)C(t+\\tau)D(t)\\right>` using the quantum
    regression theorem and the solver indicated by the 'solver' parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    tlist : *list* / *array*
        list of times for :math:`t`.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        operator A.

    b_op : :class:`qutip.qobj.Qobj`
        operator B.

    c_op : :class:`qutip.qobj.Qobj`
        operator C.

    d_op : :class:`qutip.qobj.Qobj`
        operator D.

    solver : str
        choice of solver (currently only `me` for master-equation)

    Returns
    -------

    corr_mat: *array*
        An 2-dimensional *array* (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional *array* of correlation values
        is returned instead.

    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.1.

    """

    if debug:
        print(inspect.stack()[0][3])

    if solver == "me":
        return _correlation_me_4op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, c_op, d_op,
                                      args=args, options=options)
    else:
        raise NotImplementedError("Unrecognized choice of solver %s." % solver)


#------------------------------------------------------------------------------
# high-level correlation function
#------------------------------------------------------------------------------

def coherence_function_g1(H, rho0, taulist, c_ops, a_op, solver="me",
                          args=None, options=Odeoptions()):
    """
    Calculate the first-order quantum coherence function:

    .. math::

        g^{(1)}(\\tau) = \\frac{\\langle a^\\dagger(\\tau)a(0)\\rangle}
        {\sqrt{\langle a^\dagger(\\tau)a(\\tau)\\rangle
        \\langle a^\\dagger(0)a(0)\\rangle}}

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        The annihilation operator of the mode.

    solver : str
        choice of solver ('me', 'mc', 'es')

    Returns
    -------

    g1, G2: tuble of *array*
        The normalized and unnormalized first-order coherence function.

    """

    # first calculate the photon number
    if rho0 is None:
        rho0 = steadystate(H, c_ops)
        n = np.array([expect(rho0, a_op.dag() * a_op)])
    else:
        n = mesolve(H, rho0, taulist, c_ops, [a_op.dag() * a_op],
                    args=args, options=options).expect[0]

    # calculate the correlation function G1 and normalize with n to obtain g1
    G1 = correlation_2op_1t(H, rho0, taulist, c_ops, a_op.dag(), a_op,
                            args=args, solver=solver, options=options)
    g1 = G1 / sqrt(n[0] * n)

    return g1, G1


def coherence_function_g2(H, rho0, taulist, c_ops, a_op, solver="me",
                          args=None, options=Odeoptions()):
    """
    Calculate the second-order quantum coherence function:

    .. math::

        g^{(2)}(\\tau) =
        \\frac{\\langle a^\\dagger(0)a^\\dagger(\\tau)a(\\tau)a(0)\\rangle}
        {\\langle a^\\dagger(\\tau)a(\\tau)\\rangle
         \\langle a^\\dagger(0)a(0)\\rangle}

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        The annihilation operator of the mode.

    solver : str
        choice of solver (currently only 'me')

    Returns
    -------

    g2, G2: tuble of *array*
        The normalized and unnormalized second-order coherence function.

    """

    # first calculate the photon number
    if rho0 is None:
        rho0 = steadystate(H, c_ops)
        n = np.array([expect(rho0, a_op.dag() * a_op)])
    else:
        n = mesolve(H, rho0, taulist, c_ops, [a_op.dag() * a_op], args=args).expect[0]

    # calculate the correlation function G2 and normalize with n to obtain g2
    G2 = correlation_4op_1t(H, rho0, taulist, c_ops,
                            a_op.dag(), a_op.dag(), a_op, a_op,
                            solver=solver, args=args, options=options)
    g2 = G2 / (n[0] * n)

    return g2, G2


#------------------------------------------------------------------------------
# LEGACY API
#------------------------------------------------------------------------------

def correlation_ss(H, taulist, c_ops, a_op, b_op, rho0=None, solver="me",
                   reverse=False, args=None, options=Odeoptions()):
    """
    Calculate a two-operator two-time correlation function
    :math:`\left<A(\\tau)B(0)\\right>` or
    :math:`\left<A(0)B(\\tau)\\right>` (if `reverse=True`),
    using the quantum regression theorem and the evolution solver indicated by
    the *solver* parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        operator A.

    b_op : :class:`qutip.qobj.Qobj`
        operator B.

    reverse : bool
        If `True`, calculate :math:`\left<A(0)B(\\tau)\\right>` instead of
        :math:`\left<A(\\tau)B(0)\\right>`.

    solver : str
        choice of solver (`me` for master-equation,
        `es` for exponential series and `mc` for Monte-carlo)

    Returns
    -------

    corr_vec: *array*
        An *array* of correlation values for the times specified by `tlist`

    """

    if debug:
        print(inspect.stack()[0][3])

    return correlation_2op_1t(H, rho0, taulist, c_ops, a_op, b_op,
                              solver, reverse=reverse, args=args, options=options)


def correlation(H, rho0, tlist, taulist, c_ops, a_op, b_op, solver="me",
                reverse=False, args=None, options=Odeoptions()):
    """
    Calculate a two-operator two-time correlation function on the form
    :math:`\left<A(t+\\tau)B(t)\\right>` or
    :math:`\left<A(t)B(t+\\tau)\\right>` (if `reverse=True`), using the
    quantum regression theorem and the evolution solver indicated by the
    *solver* parameter.


    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    tlist : *list* / *array*
        list of times for :math:`t`.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj`
        operator A.

    b_op : :class:`qutip.qobj`
        operator B.

    solver : str
        choice of solver (`me` for master-equation,
        `es` for exponential series and `mc` for Monte-carlo)

    Returns
    -------

    corr_mat: *array*
        An 2-dimensional *array* (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional *array* of correlation values
        is returned instead.

    """

    if debug:
        print(inspect.stack()[0][3])

    return correlation_2op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op,
                              solver=solver, reverse=reverse, args=args, options=options)


# -----------------------------------------------------------------------------
# EXPONENTIAL SERIES SOLVERS
# -----------------------------------------------------------------------------
def _correlation_es_2op_1t(H, rho0, tlist, c_ops, a_op, b_op, reverse=False,
                           args=None, options=Odeoptions()):
    """
    Internal function for calculating correlation functions using the
    exponential series solver. See :func:`correlation_ss` usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    # contruct the Liouvillian
    L = liouvillian(H, c_ops)

    # find the steady state
    if rho0 is None:
        rho0 = steadystate(L)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    # evaluate the correlation function
    if reverse:
        # <A(t)B(t+tau)>
        solC_tau = ode2es(L, rho0 * a_op)
        return esval(expect(b_op, solC_tau), tlist)
    else:
        # default: <A(t+tau)B(t)>
        solC_tau = ode2es(L, b_op * rho0)
        return esval(expect(a_op, solC_tau), tlist)


def _correlation_es_2op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op,
                           reverse=False, args=None, options=Odeoptions()):
    """
    Internal function for calculating correlation functions using the
    exponential series solver. See :func:`correlation` usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    # contruct the Liouvillian
    L = liouvillian(H, c_ops)

    if rho0 is None:
        rho0 = steadystate(L)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    solES_t = ode2es(L, rho0)

    # evaluate the correlation function
    if reverse:
        # <A(t)B(t+tau)>
        for t_idx in range(len(tlist)):
            rho_t = esval(solES_t, [tlist[t_idx]])
            solES_tau = ode2es(L, rho_t * a_op)
            C_mat[t_idx, :] = esval(expect(b_op, solES_tau), taulist)

    else:
        # default: <A(t+tau)B(t)>
        for t_idx in range(len(tlist)):
            rho_t = esval(solES_t, [tlist[t_idx]])
            solES_tau = ode2es(L, b_op * rho_t)
            C_mat[t_idx, :] = esval(expect(a_op, solES_tau), taulist)

    return C_mat


# -----------------------------------------------------------------------------
# MASTER EQUATION SOLVERS
# -----------------------------------------------------------------------------

def _correlation_me_2op_1t(H, rho0, tlist, c_ops, a_op, b_op, reverse=False,
                           args=None, options=Odeoptions()):
    """
    Internal function for calculating correlation functions using the master
    equation solver. See :func:`correlation_ss` for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    if reverse:
        # <A(t)B(t+tau)>
        return mesolve(H, rho0 * a_op, tlist, c_ops, [b_op],
                       args=args, options=options).expect[0]
    else:
        # <A(t+tau)B(t)>
        return mesolve(H, b_op * rho0, tlist, c_ops, [a_op],
                       args=args, options=options).expect[0]


def _correlation_me_2op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op,
                           reverse=False, args=None, options=Odeoptions()):
    """
    Internal function for calculating correlation functions using the master
    equation solver. See :func:`correlation` for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    rho_t_list = mesolve(H, rho0, tlist, c_ops, [], args=args, options=options).states

    if reverse:
        # <A(t)B(t+tau)>
        for t_idx, rho_t in enumerate(rho_t_list):
            C_mat[t_idx, :] = mesolve(H, rho_t * a_op, taulist,
                                      c_ops, [b_op], args=args, options=options).expect[0]
    else:
        # <A(t+tau)B(t)>
        for t_idx, rho_t in enumerate(rho_t_list):
            C_mat[t_idx, :] = mesolve(H, b_op * rho_t, taulist,
                                      c_ops, [a_op], args=args, options=options).expect[0]

    return C_mat


def _correlation_me_4op_1t(H, rho0, tlist, c_ops, a_op, b_op, c_op, d_op,
                           args=None, options=Odeoptions()):
    """
    Calculate the four-operator two-time correlation function on the form
    <A(0)B(tau)C(tau)D(0)>.

    See, Gardiner, Quantum Noise, Section 5.2.1
    """

    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    return mesolve(H, d_op * rho0 * a_op, tlist,
                   c_ops, [b_op * c_op], args=args, options=options).expect[0]


def _correlation_me_4op_2t(H, rho0, tlist, taulist, c_ops,
                           a_op, b_op, c_op, d_op, reverse=False,
                           args=None, options=Odeoptions()):
    """
    Calculate the four-operator two-time correlation function on the form
    <A(t)B(t+tau)C(t+tau)D(t)>.

    See, Gardiner, Quantum Noise, Section 5.2.1
    """

    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    rho_t = mesolve(H, rho0, tlist, c_ops, [], args=args, options=options).states

    for t_idx, rho in enumerate(rho_t):
        C_mat[t_idx, :] = mesolve(H, d_op * rho * a_op, taulist,
                                  c_ops, [b_op * c_op],
                                  args=args, options=options).expect[0]

    return C_mat


# -----------------------------------------------------------------------------
# MONTE CARLO SOLVERS
# -----------------------------------------------------------------------------
def _correlation_mc_2op_1t(H, psi0, taulist, c_ops, a_op, b_op, reverse=False,
                           args=None, options=Odeoptions()):
    """
    Internal function for calculating correlation functions using the Monte
    Carlo solver. See :func:`correlation_ss` for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    if psi0 is None or not isket(psi0):
        raise Exception("_correlation_mc_2op_1t requires initial state as ket")

    options.gui = False

    b_op_psi0 = b_op * psi0

    norm = b_op_psi0.norm()

    return norm * mcsolve(H, b_op_psi0 / norm, taulist, c_ops, [a_op],
                          args=args, options=options).expect[0]


def _correlation_mc_2op_2t(H, psi0, tlist, taulist, c_ops, a_op, b_op,
                           reverse=False, args=None, options=Odeoptions()):
    """
    Internal function for calculating correlation functions using the Monte
    Carlo solver. See :func:`correlation` usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    raise NotImplementedError("The Monte-Carlo solver currently cannot be " +
                              "used for correlation functions on the form " +
                              "<A(t)B(t+tau)>")

    if psi0 is None or not isket(psi0):
        raise Exception("_correlation_mc_2op_2t requires initial state as ket")

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    options.gui = False

    psi_t = mcsolve(H, psi0, tlist, c_ops, [], args=args, options=options).states

    for t_idx in range(len(tlist)):

        psi0_t = psi_t[0][t_idx]

        C_mat[t_idx, :] = mcsolve(H, b_op * psi0_t, tlist, c_ops, [a_op],
                                  args=args, options=options).expect[0]

    return C_mat


# -----------------------------------------------------------------------------
# SPECTRUM
# -----------------------------------------------------------------------------
def spectrum_correlation_fft(tlist, y):
    """
    Calculate the power spectrum corresponding to a two-time correlation
    function using FFT.

    Parameters
    ----------

    tlist : *list* / *array*
        list/array of times :math:`t` which the correlation function is given.

    y : *list* / *array*
        list/array of correlations corresponding to time delays :math:`t`.

    Returns
    -------

    w, S : *tuple*

        Returns an array of angular frequencies 'w' and the corresponding
        one-sided power spectrum 'S(w)'.

    """

    if debug:
        print(inspect.stack()[0][3])

    N = len(tlist)
    dt = tlist[1] - tlist[0]

    F = scipy.fftpack.fft(y)

    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)

    # select only indices for elements that corresponds
    # to positive frequencies
    indices = np.where(f > 0.0)

    return 2 * pi * f[indices], 2 * dt * np.real(F[indices])


def spectrum_ss(H, wlist, c_ops, a_op, b_op):
    """
    Calculate the spectrum corresponding to a correlation function
    :math:`\left<A(\\tau)B(0)\\right>`, i.e., the Fourier transform of the
    correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty} \left<A(\\tau)B(0)\\right>
        e^{-i\omega\\tau} d\\tau.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    wlist : *list* / *array*
        list of frequencies for :math:`\\omega`.

    c_ops : list of :class:`qutip.qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj`
        operator A.

    b_op : :class:`qutip.qobj`
        operator B.

    Returns
    -------

    spectrum: *array*
        An *array* with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """

    if debug:
        print(inspect.stack()[0][3])

    # contruct the Liouvillian
    L = liouvillian(H, c_ops)

    # find the steady state density matrix and a_op and b_op expecation values
    rho0 = steadystate(L)

    a_op_ss = expect(a_op, rho0)
    b_op_ss = expect(b_op, rho0)

    # eseries solution for (b * rho0)(t)
    es = ode2es(L, b_op * rho0)

    # correlation
    corr_es = expect(a_op, es)

    # covarience
    cov_es = corr_es - np.real(np.conjugate(a_op_ss) * b_op_ss)

    # spectrum
    spectrum = esspec(cov_es, wlist)

    return spectrum

	
def spectrum_pi(L, aop, bop, wlist, use_pinv=False):
    """
    Calculate the spectrum corresponding to a correlation function
    :math:`\left<A(\\tau)B(0)\\right>`, i.e., the Fourier transform of the
    correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty} \left<A(\\tau)B(0)\\right>
        e^{-i\omega\\tau} d\\tau.

    Parameters
    ----------

    L : :class:`qutip.qobj`
        system Liouvillian.

    wlist : *list* / *array*
        list of frequencies for :math:`\\omega`.

    
    a_op : :class:`qutip.qobj`
        operator A.

    b_op : :class:`qutip.qobj`
        operator B.

    Returns
    -------

    s_vec: *array*
        An *array* with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """

    unitvec = tensor([qeye(n) for n in L.dims[0][0]])
    N = prod(L.dims[0][0])
    
    A = L.full()
    aflat = spre(bop).full()
    adagflat = spre(aop).full()
    
    unitflat = transpose(mat2vec(unitvec.full()))
    I = np.identity(N * N)

    rho_ss = steady(L)    
    rho = transpose(mat2vec(rho_ss.full()))

    P = np.kron(transpose(rho), unitflat)
    Q = I - P

    s_vec = zeros(len(wlist))
    
    for idx, freq in enumerate(wlist):
        
        if use_pinv:
            MMR=numpy.linalg.pinv(-freq*1.0j*unit22+A)
        else:
            MMR = dot(Q, np.linalg.solve((-freq*1.0j*I+A),Q))
        
        
        s_vec[idx] = np.real(-2*dot(unitflat,dot(adagflat,dot(MMR,dot(aflat,transpose(rho)))))[0,0])

    return s_vec

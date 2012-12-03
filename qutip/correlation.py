#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from qutip.superoperator import *
from qutip.expect import expect
from qutip.mesolve import mesolve
from qutip.eseries import esval, esspec
from qutip.essolve import ode2es
from qutip.mcsolve import mcsolve
from qutip.steady import steady, steadystate
import numpy as np


#------------------------------------------------------------------------------
# solver wrapers:
#
def correlation_ss(H, tlist, c_op_list, a_op, b_op, rho0=None, solver="me"):
    """
    Calculate a two-time correlation function
    :math:`\left<A(\\tau)B(0)\\right>` using the quantum regression
    theorem, using the solver indicated by the *solver* parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_op_list : list of :class:`qutip.qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj`
        operator A.

    b_op : :class:`qutip.qobj`
        operator B.

    rho0 : :class:`qutip.qobj`
        Optional initial state density matrix (default is the steady state).

    solver : str
        choice of solver (`me` for master-equation,
        `es` for exponential series and `mc` for Monte-carlo)

    Returns
    -------

    corr_list: *array*
        An *array* of correlation values for the times specified by `tlist`

    """

    if solver == "me":
        return correlation_ss_ode(H, tlist, c_op_list, a_op, b_op, rho0)
    elif solver == "es":
        return correlation_ss_es(H, tlist, c_op_list, a_op, b_op, rho0)
    elif solver == "mc":
        print("Monte-Carlo solver is currently disabled, " +
              "using master equation.")
        return correlation_ss_ode(H, tlist, c_op_list, a_op, b_op, rho0)
    else:
        raise "Unrecognized choice of solver %s (use me, es or mc)." % solver


def correlation(H, rho0, tlist, taulist, c_op_list, a_op, b_op, solver="me"):
    """
    Calculate a two-time correlation function
    :math:`\left<A(t+\\tau)B(t)\\right>` using the quantum regression
    theorem, using the solver indicated by the *solver* parameter.


    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj`
        initial density matrix :math:`\\rho(t_0)`

    tlist : *list* / *array*
        list of times for :math:`t`.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_op_list : list of :class:`qutip.qobj`
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

    if tlist == None:
        # only interested in correlation vs one time coordinate, so we can use
        # the ss solver with the supplied density matrix as initial state (in
        # place of the steady state)
        return correlation_ss(H, taulist, c_op_list, a_op, b_op, rho0, solver)

    if solver == "me":
        return correlation_ode(H, rho0, tlist, taulist, c_op_list, a_op, b_op)
    elif solver == "es":
        return correlation_es(H, rho0, tlist, taulist, c_op_list, a_op, b_op)
    elif solver == "es":
        return correlation_mc(H, rho0, tlist, taulist, c_op_list, a_op, b_op)
    else:
        raise "Unrecognized choice of solver %s (use me, es or mc)." % solver


def correlation_ss_gtt(H, tlist, c_ops, a_op, b_op, c_op, d_op, rho0=None,
                       solver="me"):
    """
    Calculate the correlation function <A(0)B(tau)C(tau)D(0)>

    (ss_gtt = steadystate general two-time)

    See, Gardiner, Quantum Noise, Section 5.2.1

    .. note::
        Experimental.
    """

    if solver == "me":
        return _correlation_me_ss_gtt(H, tlist, c_ops, a_op, b_op, c_op,
                                      d_op, rho0)
    else:
        raise NotImplementedError("Unrecognized choice of solver %s." % solver)

# -----------------------------------------------------------------------------
# EXPONENTIAL SERIES SOLVERS
# -----------------------------------------------------------------------------


def correlation_ss_es(H, tlist, c_op_list, a_op, b_op, rho0=None):
    """
    Internal function for calculating correlation functions using the
    exponential series solver. See :func:`correlation_ss` usage.
    """

    # contruct the Liouvillian
    L = liouvillian(H, c_op_list)

    # find the steady state
    if rho0 == None:
        rho0 = steady(L)

    # evaluate the correlation function
    solC_tau = ode2es(L, b_op * rho0)

    return esval(expect(a_op, solC_tau), tlist)


def correlation_es(H, rho0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Internal function for calculating correlation functions using the
    exponential series solver. See :func:`correlation` usage.
    """

    # contruct the Liouvillian
    L = liouvillian(H, c_op_list)

    if rho0 == None:
        rho0 = steady(L)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    solES_t = ode2es(L, rho0)

    for t_idx in range(len(tlist)):

        rho_t = esval(solES_t, [tlist[t_idx]])

        solES_tau = ode2es(L, b_op * rho_t)

        C_mat[t_idx, :] = esval(expect(a_op, solES_tau), taulist)

    return C_mat

# -----------------------------------------------------------------------------
# MASTER EQUATION SOLVERS
# -----------------------------------------------------------------------------


def correlation_ss_ode(H, tlist, c_op_list, a_op, b_op, rho0=None):
    """
    Internal function for calculating correlation functions using the master
    equation solver. See :func:`correlation_ss` usage.
    """

    L = liouvillian(H, c_op_list)
    if rho0 == None:
        rho0 = steady(L)

    return mesolve(H, b_op * rho0, tlist, c_op_list, [a_op]).expect[0]


def _correlation_me_ss_gtt(H, tlist, c_ops, a_op, b_op, c_op, d_op, rho0=None):
    """
    Calculate the correlation function <A(0)B(tau)C(tau)D(0)>

    (ss_gtt = steadystate general two-time)

    See, Gardiner, Quantum Noise, Section 5.2.1

    .. note::
        Experimental.
    """
    if rho0 is None:
        rho0 = steadystate(H, c_ops)

    return mesolve(H, d_op * rho0 * a_op, tlist,
                   c_ops, [b_op * c_op]).expect[0]


def correlation_ode(H, rho0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Internal function for calculating correlation functions using the master
    equation solver. See :func:`correlation` usage.
    """

    if rho0 == None:
        rho0 = steadystate(H, co_op_list)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    rho_t = mesolve(H, rho0, tlist, c_op_list, []).states

    for t_idx in range(len(tlist)):
        C_mat[t_idx, :] = mesolve(H, b_op * rho_t[t_idx], taulist,
                                  c_op_list, [a_op]).expect[0]

    return C_mat


def _correlation_me_gtt(H, rho0, tlist, taulist, c_ops, a_op, b_op,
                        c_op, d_op):
    """
    Calculate the correlation function <A(t)B(t+tau)C(t+tau)D(t)>

    (gtt = general two-time)

    See, Gardiner, Quantum Noise, Section 5.2.1

    .. note::
        Experimental.
    """
    if rho0 is None:
        rho0 = steadystate(H, c_ops)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    rho_t = mesolve(H, rho0, tlist, c_op_list, []).states

    for t_idx, rho in enumerate(rho_t):
        C_mat[t_idx, :] = mesolve(H, d_op * rho * a_op, taulist,
                                 c_ops, [b_op * c_op]).expect[0]

    return C_mat

# -----------------------------------------------------------------------------
# MONTE CARLO SOLVERS
# -----------------------------------------------------------------------------


def correlation_ss_mc(H, tlist, c_op_list, a_op, b_op, rho0=None):
    """
    Internal function for calculating correlation functions using the Monte
    Carlo solver. See :func:`correlation_ss` usage.
    """

    if rho0 == None:
        rho0 = steadystate(H, co_op_list)

    return mcsolve(H, b_op * rho0, tlist, c_op_list, [a_op]).expect[0]


def correlation_mc(H, psi0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Internal function for calculating correlation functions using the Monte
    Carlo solver. See :func:`correlation` usage.
    """

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    ntraj = 100

    mc_opt = Mcoptions()
    mc_opt.progressbar = False

    psi_t = mcsolve(H, psi0, tlist, ntraj, c_op_list, [], mc_opt).states

    for t_idx in range(len(tlist)):

        psi0_t = psi_t[0][t_idx]

        C_mat[t_idx, :] = mcsolve(H, b_op * psi0_t, tlist,
                                  ntraj, c_op_list, [a_op], mc_opt).expect[0]

    return C_mat


# -----------------------------------------------------------------------------
# SPECTRUM
# -----------------------------------------------------------------------------

def spectrum_ss(H, wlist, c_op_list, a_op, b_op):
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

    c_op_list : list of :class:`qutip.qobj`
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

    # contruct the Liouvillian
    L = liouvillian(H, c_op_list)

    # find the steady state density matrix and a_op and b_op expecation values
    rho0 = steady(L)

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

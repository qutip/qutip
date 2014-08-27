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

import numpy as np
import scipy.fftpack
import scipy.interpolate
import os, sys

from qutip.superoperator import *
from qutip.expect import expect
from qutip.tensor import tensor
from qutip.operators import qeye
from qutip.mesolve import mesolve
from qutip.eseries import esval, esspec
from qutip.essolve import ode2es
from qutip.mcsolve import mcsolve
from qutip.steadystate import steadystate
from qutip.states import ket2dm
from qutip.solver import Options
from qutip.settings import debug
from re import sub

if debug:
    import inspect


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def correlation_2op_1t(H, rho0, taulist, c_ops, a_op, b_op, solver="me",
                       reverse=False, args=None, options=Options()):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\\tau)B(t)\\right>`
    along one time axis using the quantum regression theorem and the evolution
    evolution solver indicated by the *solver* parameter.

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
        If `True`, calculate :math:`\left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\left<A(t+\\tau)B(t)\\right>`.

    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series)

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
                                      reverse)
    else:
        raise "Unrecognized choice of solver %s (me, es or mc)." % solver


def correlation_2op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op, solver="me",
                       reverse=False, args=None, options=Options()):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\\tau)B(t)\\right>`
    along two time axes using the quantum regression theorem and the evolution
    evolution solver indicated by the *solver* parameter.


    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian, or a callback function for time dependent
        Hamiltonian IF using the me solver.

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
        choice of solver (`me` for master-equation and
        `es` for exponential series)

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
        # the 1t solver with the supplied density matrix as initial state
        return correlation_2op_1t(H, rho0, taulist, c_ops, a_op, b_op, solver,
                                  reverse, args=args, options=options)

    if solver == "me":
        return _correlation_me_2op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, reverse, args=args,
                                      options=options)
    elif solver == "es":
        return _correlation_es_2op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, reverse)
    else:
        raise "Unrecognized choice of solver %s (use me or es)." % solver


def correlation_4op_1t(H, rho0, taulist, c_ops, a_op, b_op, c_op, d_op,
                       solver="me", args=None, options=Options()):
    """
    Calculate the four-operator two-time correlation function:
    :math:`\left<A(t)B(t+\\tau)C(t+\\tau)D(t)\\right>`
    along one time axis using the quantum regression theorem and the
    evolution solver indicated by the *solver* parameter.

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

    # Note: reverse is not sufficient for all possible normal and correctly
    # time ordered version of the 4 operators. It seems least cluttered to
    # leave such an argument out and simply have the user input the correct
    # orderings when needed.

    if debug:
        print(inspect.stack()[0][3])

    if solver == "me":
        return _correlation_me_4op_1t(H, rho0, taulist, c_ops,
                                      a_op, b_op, c_op, d_op,
                                      args=args, options=options)
    else:
        raise NotImplementedError("Unrecognized choice of solver %s." % solver)


def correlation_4op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op, c_op, d_op,
                       solver="me", args=None, options=Options()):
    """
    Calculate the four-operator two-time correlation function:
    :math:`\left<A(t)B(t+\\tau)C(t+\\tau)D(t)\\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the *solver* parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian, or a callback function for time-dependent
        Hamiltonians.

    rho0 : :class:`qutip.qobj.Qobj`
        Initial state density matrix (or state vector). If 'rho0' is
        'None', then the steady state will be used as initial state.

    tlist : *list* / *array*
        list of times for :math:`t`.

    taulist : *list* / *array*
        list of times for :math:`\\tau`. This list can include negative
        values.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators. (does not accept time dependence)

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

    # Note: reverse is not sufficient for all possible normal and correctly
    # time ordered version of the 4 operators. It seems least cluttered to
    # leave such an argument out and simply have the user input the correct
    # orderings when needed.

    if debug:
        print(inspect.stack()[0][3])

    if tlist is None:
        # only interested in correlation vs one time coordinate, so we can use
        # the 1t solver with the supplied density matrix as initial state
        return correlation_4op_1t(H, rho0, taulist, c_ops, a_op, b_op, solver,
                                  args=args, options=options)

    if solver == "me":
        return _correlation_me_4op_2t(H, rho0, tlist, taulist, c_ops,
                                      a_op, b_op, c_op, d_op,
                                      args=args, options=options)
    else:
        raise NotImplementedError("Unrecognized choice of solver %s." % solver)


# -----------------------------------------------------------------------------
# LEGACY API
# -----------------------------------------------------------------------------

def correlation_ss(H, taulist, c_ops, a_op, b_op, solver="me",
                   reverse=False, args=None, options=Options()):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(\\tau)B(0)\\right>`
    along one time axis using the quantum regression theorem and the
    evolution solver indicated by the *solver* parameter.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

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
        choice of solver (`me` for master-equation and
        `es` for exponential series)

    Returns
    -------

    corr_vec: *array*
        An *array* of correlation values for the times specified by `tlist`

    """

    if debug:
        print(inspect.stack()[0][3])

    return correlation_2op_1t(H, None, taulist, c_ops, a_op, b_op,
                              solver, reverse=reverse, args=args,
                              options=options)


def correlation(H, rho0, tlist, taulist, c_ops, a_op, b_op, solver="me",
                reverse=False, args=None, options=Options()):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\\tau)B(t)\\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the *solver* parameter.


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

    reverse : bool
        If `True`, calculate :math:`\left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\left<A(t+\\tau)B(t)\\right>`.

    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series)

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
                              solver=solver, reverse=reverse, args=args,
                              options=options)


# -----------------------------------------------------------------------------
# HIGH-LEVEL CORRELATION FUNCTIONS
# -----------------------------------------------------------------------------

def coherence_function_g1(H, taulist, c_ops, a_op, solver="me",
                          args=None, options=Options()):
    """
    Calculate the normalized first-order quantum coherence function:

    .. math::

        g^{(1)}(\\tau) = \\frac{\\langle a^\\dagger(\\tau)a(0)\\rangle}
        {\sqrt{\langle a^\dagger(\\tau)a(\\tau)\\rangle
        \\langle a^\\dagger(0)a(0)\\rangle}}

    using the quantum regression theorem and the evolution solver indicated by
    the *solver* parameter. Note: g1 is only defined for stationary
    statistics (uses steady state rho0)

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        The annihilation operator of the mode.

    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series)

    Returns
    -------

    g1, G2: tuble of *array*
        The normalized first-order coherence function.

    """

    # first calculate the photon number
    rho0 = steadystate(H, c_ops)
    n = np.array([expect(rho0, a_op.dag() * a_op)])

    # calculate the correlation function G1 and normalize with n to obtain g1
    G1 = correlation_2op_1t(H, None, taulist, c_ops, a_op.dag(), a_op,
                            args=args, solver=solver, options=options)
    g1 = G1 / sqrt(n[0] * n)

    return g1, G1


def coherence_function_g2(H, taulist, c_ops, a_op, solver="me",
                          args=None, options=Options()):
    """
    Calculate the normalized second-order quantum coherence function:

    .. math::

        g^{(2)}(\\tau) =
        \\frac{\\langle a^\\dagger(0)a^\\dagger(\\tau)a(\\tau)a(0)\\rangle}
        {\\langle a^\\dagger(\\tau)a(\\tau)\\rangle
         \\langle a^\\dagger(0)a(0)\\rangle}

    using the quantum regression theorem and the evolution solver indicated by
    the *solver* parameter. Note: g2 is only defined for stationary
    statistics (uses steady state rho0)

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        system Hamiltonian.

    taulist : *list* / *array*
        list of times for :math:`\\tau`.

    c_ops : list of :class:`qutip.qobj.Qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj.Qobj`
        The annihilation operator of the mode.

    solver : str
        choice of solver (currently only `me` for master-equation)

    Returns
    -------

    g2, G2: tuble of *array*
        The normalized and unnormalized second-order coherence function.

    """

    # first calculate the photon number
    rho0 = steadystate(H, c_ops)
    n = np.array([expect(rho0, a_op.dag() * a_op)])

    # calculate the correlation function G2 and normalize with n to obtain g2
    G2 = correlation_4op_1t(H, None, taulist, c_ops,
                            a_op.dag(), a_op.dag(), a_op, a_op,
                            solver=solver, args=args, options=options)
    g2 = G2 / (n[0] * n)

    return g2, G2


# -----------------------------------------------------------------------------
# SPECTRUM
# -----------------------------------------------------------------------------

def spectrum(H, wlist, c_ops, a_op, b_op, solver='es', use_pinv=False):
    """
    Calculate the spectrum corresponding to the correlation function
    :math:`\left<A(\\tau)B(0)\\right>`, i.e., the Fourier transform of the
    correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty} \left<A(\\tau)B(0)\\right>
        e^{-i\omega\\tau} d\\tau.

    using the solver indicated by the *solver* parameter. Note: this spectrum
    is only defined for stationary statistics (uses steady state rho0)

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

    solver : str
        choice of solver (`es` for exponential series and
        `pi` for psuedo-inverse)

    use_pinv : bool
        If `True` use numpy's pinv method, otherwise use a generic solver

    Returns
    -------

    spectrum: *array*
        An *array* with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """

    if debug:
        print(inspect.stack()[0][3])

    if solver == "es":
        return _spectrum_es(H, wlist, c_ops, a_op, b_op)
    elif solver == "pi":
        return _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv)
    else:
        raise NotImplementedError("Unrecognized choice of solver %s." % solver)


def spectrum_correlation_fft(taulist, y):
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

    N = len(taulist)
    dt = taulist[1] - taulist[0]

    F = scipy.fftpack.fft(y)

    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)

    # select only indices for elements that corresponds
    # to positive frequencies
    indices = np.where(f > 0.0)

    return 2 * pi * f[indices], 2 * dt * np.real(F[indices])


# -----------------------------------------------------------------------------
# EXPONENTIAL SERIES SOLVERS
# -----------------------------------------------------------------------------

def _correlation_es_2op_1t(H, rho0, taulist, c_ops, a_op, b_op, reverse=False):
    """
    Internal function for calculating the two-operator two-time
    correlation function:
    <A(t)B(t+tau)>/<A(t+tau)B(t)>
    along one time axis using an exponential series solver.
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
        return esval(expect(b_op, solC_tau), taulist)
    else:
        # default: <A(t+tau)B(t)>
        solC_tau = ode2es(L, b_op * rho0)
        return esval(expect(a_op, solC_tau), taulist)


def _correlation_es_2op_2t(H, rho0, tlist, taulist,
                           c_ops, a_op, b_op, reverse=False):
    """
    Internal function for calculating the two-operator two-time
    correlation function:
    <A(t)B(t+tau)>/<A(t+tau)B(t)>
    along two time axes using an exponential series solver.
    """

    if min(taulist) < 0:
        raise TypeError("The quantum regression theorem can only be used" +
                        "in calculating correlations with a positive time" +
                        "difference, tau")
    if debug:
        print(inspect.stack()[0][3])

    # contruct the Liouvillian
    L = liouvillian(H, c_ops)

    if rho0 is None:
        rho0 = steadystate(L)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    solES_t = ode2es(L, rho0)

    # evaluate the correlation function
    if reverse:
        # <A(t)B(t+tau)>
        for t_idx in range(len(tlist)):
            rho_t = esval(solES_t, [tlist[t_idx]])
            solES_tau = ode2es(L, rho_t * a_op)
            corr_mat[t_idx, :] = esval(expect(b_op, solES_tau), taulist)

    else:
        # <A(t+tau)B(t)>
        for t_idx in range(len(tlist)):
            rho_t = esval(solES_t, [tlist[t_idx]])
            solES_tau = ode2es(L, b_op * rho_t)
            corr_mat[t_idx, :] = esval(expect(a_op, solES_tau), taulist)

    return corr_mat


def _spectrum_es(H, wlist, c_ops, a_op, b_op):
    """
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\\tau)B(0)\\right>`.
    """
    if debug:
        print(inspect.stack()[0][3])

    # construct the Liouvillian
    L = liouvillian(H, c_ops)

    # find the steady state density matrix and a_op and b_op expecation values
    rho0 = steadystate(L)

    a_op_ss = expect(a_op, rho0)
    b_op_ss = expect(b_op, rho0)

    # eseries solution for (b * rho0)(t)
    es = ode2es(L, b_op * rho0)

    # correlation
    corr_es = expect(a_op, es)

    # covariance
    cov_es = corr_es - np.real(np.conjugate(a_op_ss) * b_op_ss)

    # spectrum
    spectrum = esspec(cov_es, wlist)

    return spectrum


# -----------------------------------------------------------------------------
# MASTER EQUATION SOLVERS
# -----------------------------------------------------------------------------

# See, Gardiner, Quantum Noise, Section 5.2.1

def _correlation_me_2op_1t(H, rho0, taulist, c_ops, a_op, b_op,
                           reverse=False, args=None, options=Options()):
    """
    Internal function for calculating the two-operator two-time
    correlation function:
    <A(t)B(t+tau)>/<A(t+tau)B(t)>
    along one time axis using a master equation based solver.
    """

    if min(taulist) < 0:
        raise TypeError("The quantum regression theorem can only be used" +
                        "in calculating correlations with a positive time" +
                        "difference, tau")
    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    if reverse:
        # <A(0)B(tau)>
        correlator_ic = rho0 * a_op
        correlator_exp = b_op
    else:
        # <A(tau)B(0)>
        correlator_ic = b_op * rho0
        correlator_exp = a_op

    return mesolve(H, correlator_ic, taulist, c_ops, [correlator_exp],
                       args=args, options=options).expect[0]


def _correlation_me_2op_2t(H, rho0, tlist, taulist, c_ops, a_op, b_op,
                           reverse=False, args=None, options=Options()):
    """
    Internal function for calculating the two-operator two-time
    correlation function:
    <A(t)B(t+tau)>/<A(t+tau)B(t)>
    along two time axes using a master equation based solver.
    """

    if min(taulist) < 0:
        raise TypeError("The quantum regression theorem can only be used" +
                        "in calculating correlations with a positive time" +
                        "difference, tau")
    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None and not isinstance(H, list):
        rho0 = steadystate(H, c_ops)
    elif rho0 is None:
        raise TypeError("Missing initial condition")
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    rho_t = mesolve(H, rho0, tlist, c_ops, [],
                    args=args, options=options).states
    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    H_shifted, _args = _transform_H_t_shift(H, args)

    for t_idx, rho in enumerate(rho_t):
        if not isinstance(H, Qobj):
            _args["_t0"] = tlist[t_idx]

        if reverse:
            # <A(t)B(t+tau)>
            correlator_ic = rho * a_op
            correlator_exp = b_op
        else:
            # <A(t+tau)B(t)>
            correlator_ic = b_op * rho
            correlator_exp = a_op

        corr_mat[t_idx, :] = mesolve(
            H_shifted, correlator_ic, taulist, c_ops,
            [correlator_exp], args=_args, options=options
        ).expect[0]

        if t_idx == 1:
            options.rhs_reuse = True

    return corr_mat


def _correlation_me_4op_1t(H, rho0, taulist, c_ops, a_op, b_op, c_op, d_op,
                           args=None, options=Options()):
    """
    Internal function for calculating the four-operator two-time
    correlation function:
    <A(t)B(t+tau)C(t+tau)D(t)>
    along one time axis using a master equation based solver.
    """

    if min(taulist) < 0:
        raise TypeError("The quantum regression theorem can only be used" +
                        "in calculating correlations with a positive time" +
                        "difference, tau")
    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    return mesolve(H, d_op * rho0 * a_op, taulist, c_ops, [b_op * c_op],
                   args=args, options=options).expect[0]


def _correlation_me_4op_2t(H, rho0, tlist, taulist, c_ops,
                           a_op, b_op, c_op, d_op,
                           args=None, options=Options()):
    """
    Internal function for calculating the four-operator two-time
    correlation function:
    <A(t)B(t+tau)C(t+tau)D(t)>
    along two time axes using a master equation based solver.
    """

    if min(taulist) < 0:
        raise TypeError("The quantum regression theorem can only be used" +
                        "in calculating correlations with a positive time" +
                        "difference, tau")
    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None and not isinstance(H, list):
        rho0 = steadystate(H, c_ops)
    elif rho0 is None:
        raise TypeError("Missing initial condition")
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    rho_t = mesolve(H, rho0, tlist, c_ops, [],
                    args=args, options=options).states
    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    H_shifted, _args = _transform_H_t_shift(H, args)

    for t_idx, rho in enumerate(rho_t):
        if not isinstance(H, Qobj):
            _args["_t0"] = tlist[t_idx]

        corr_mat[t_idx, :] = mesolve(
            H_shifted, d_op * rho * a_op, taulist, c_ops,
            [b_op * c_op], args=_args, options=options
        ).expect[0]

        if t_idx == 1:
            options.rhs_reuse = True

    return corr_mat


# -----------------------------------------------------------------------------
# MONTE CARLO SOLVERS
# -----------------------------------------------------------------------------

def corr_mc_2op_1t_single_evolve(H, psi_t, taulist, c_ops, a_op, b_op,
                                 reverse=False, args=None, n2traj=25):
    # <A(t+tau)B(t)>/<A(t)B(t+tau)>
    # insert reference!!!

    # the mcsolve output is not very informative here
    devnull = open(os.devnull, 'w')
    sys.stdout, sys.stderr = devnull, devnull  # suppress mcsolve output

    if not isket(psi_t):
        raise TypeError("The initial condition must be a state vector.")

    if reverse:
        A_op = b_op
        B_op = a_op
        r_sgn = -1
    else:
        A_op = a_op
        B_op = b_op
        r_sgn = 1

    # generate states (1+B)*psi_t, (1+iB)*psi_t, (1-B)*psi_t, (1-iB)*psi_t
    chi_0 = [(1 + exp(1j*x*pi/2)*B_op) * psi_t for x in range(4)]

    # evolve these states and calculate expectation value of A
    c_tau = [
        chi.norm()**2 * mcsolve(
            H, chi/chi.norm(), taulist, c_ops, [A_op], args=args, ntraj=n2traj
        ).expect[0]
        for chi in chi_0
    ]

    # final correlation vector computed by combining the averages
    corr_vec = 1/4 * (c_tau[0] - c_tau[2] - r_sgn*1j*c_tau[1] + r_sgn*1j*c_tau[3])

    sys.stdout = sys.__stdout__  # return system output pointers
    sys.stderr = sys.__stderr__  # return system output pointers

    return corr_vec


def corr_mc_2op_1t_double_evolve(H, psi0, t_corr, taulist, c_ops, a_op, b_op,
                                 reverse=False, args=None, n1traj=5, n2traj=25):
    # <A(t+tau)B(t)>/<A(t)B(t+tau)>
    # insert reference!!!

    if not isket(psi0):
        raise TypeError("The initial condition must be a state vector.")

    # the mcsolve output is not very informative here
    devnull = open(os.devnull, 'w')
    sys.stdout, sys.stderr = devnull, devnull  # suppress mcsolve output

    psi_t_list = mcsolve(
        H, psi0, [0, t_corr], c_ops, [], args=args, ntraj=n1traj
    ).states[:, 1]

    corr_vec = np.zeros([np.size(taulist)], dtype=complex)

    for trial_idx in range(n1traj):
        # calculation of <A(t+tau)B(t)>/<A(t)B(t+tau)> from only knowledge of
        # phi0 requires averaging over both t and tau
        if reverse:
            A_op = b_op
            B_op = a_op
            r_sgn = -1
        else:
            A_op = a_op
            B_op = b_op
            r_sgn = 1

        # generate states (1+B)*psi_t, (1+iB)*psi_t, (1-B)*psi_t, (1-iB)*psi_t
        chi_0 = [(1 + exp(1j*x*pi/2)*B_op) * psi_t_list[trial_idx]
                 for x in range(4)]

        # evolve these states and calculate expectation value of A
        c_tau = [
            chi.norm()**2 * mcsolve(
                H, chi/chi.norm(), taulist, c_ops, [A_op],
                args=args, ntraj=n2traj
            ).expect[0]
            for chi in chi_0
        ]

        # final correlation vector computed by combining the averages
        corr_vec += 1/(4*n1traj) * (c_tau[0] - c_tau[2] -
                                    r_sgn*1j*c_tau[1] + r_sgn*1j*c_tau[3])

    sys.stdout = sys.__stdout__  # return system output pointers
    sys.stderr = sys.__stderr__  # return system output pointers

    return corr_vec


def corr_mc_2op_2t(H, psi0, tlist, taulist, c_ops, a_op, b_op,
                                 reverse=False, args=None, n1traj=3, n2traj=15):
    # <A(t+tau)B(t)>/<A(t)B(t+tau)>
    # insert reference!!!

    if not isket(psi0):
        raise TypeError("The initial condition must be a state vector.")

    # the mcsolve output is not very informative here
    devnull = open(os.devnull, 'w')
    sys.stdout, sys.stderr = devnull, devnull  # suppress mcsolve output

    psi_t_mat = mcsolve(
        H, psi0, tlist, c_ops, [], args=args, ntraj=n1traj
    ).states

    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    for t_idx in range(np.size(tlist)):
        for trial_idx in range(n1traj):
            # calculation of <A(t+tau)B(t)>/<A(t)B(t+tau)> from only knowledge of
            # phi0 requires averaging over both t and tau
            if reverse:
                A_op = b_op
                B_op = a_op
                r_sgn = -1
            else:
                A_op = a_op
                B_op = b_op
                r_sgn = 1

            # generate states (1+B)*psi_t, (1+iB)*psi_t, (1-B)*psi_t, (1-iB)*psi_t
            chi_0 = [(1 + exp(1j*x*pi/2)*B_op) * psi_t_mat[trial_idx, t_idx]
                     for x in range(4)]

            # evolve these states and calculate expectation value of A
            c_tau = [
                chi.norm()**2 * mcsolve(
                    H, chi/chi.norm(), taulist, c_ops, [A_op],
                    args=args, ntraj=n2traj
                ).expect[0]
                for chi in chi_0
            ]

            # final correlation vector computed by combining the averages
            corr_mat[t_idx,:] += 1/(4*n1traj) * (c_tau[0] - c_tau[2] -
                                        r_sgn*1j*c_tau[1] + r_sgn*1j*c_tau[3])

    sys.stdout = sys.__stdout__  # return system output pointers
    sys.stderr = sys.__stderr__  # return system output pointers

    return corr_mat


# -----------------------------------------------------------------------------
# PSUEDO-INVERSE SOLVERS
# -----------------------------------------------------------------------------

def _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv=False):
    """
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\\tau)B(0)\\right>`.
    """

    L = H if issuper(H) else liouvillian(H, c_ops)

    tr_mat = tensor([qeye(n) for n in L.dims[0][0]])
    N = prod(L.dims[0][0])

    A = L.full()
    b = spre(b_op).full()
    a = spre(a_op).full()

    tr_vec = transpose(mat2vec(tr_mat.full()))

    rho_ss = steadystate(L)
    rho = transpose(mat2vec(rho_ss.full()))

    I = np.identity(N * N)
    P = np.kron(transpose(rho), tr_vec)
    Q = I - P

    spectrum = np.zeros(len(wlist))

    for idx, w in enumerate(wlist):
        if use_pinv:
            MMR = numpy.linalg.pinv(-1.0j * w * I + A)
        else:
            MMR = np.dot(Q, np.linalg.solve(-1.0j * w * I + A, Q))

        s = np.dot(tr_vec, np.dot(a, np.dot(MMR, np.dot(b, transpose(rho)))))
        spectrum[idx] = -2 * np.real(s[0, 0])

    return spectrum


# -----------------------------------------------------------------------------
# AUXILIARY
# -----------------------------------------------------------------------------

def _transform_H_t_shift(H, args=None):
    """
    Time shift the Hamiltonian with private time-shift variable _t0
    """

    if isinstance(H, Qobj):
        # constant hamiltonian
        return H, args

    if isinstance(H, types.FunctionType):
        # function-callback based time-dependence
        if isinstance(args, dict) or args is None:
            if args is None:
                _args = {"_t0": 0}
            else:
                _args = args.copy()
                _args["_t0"] = 0
            H_shifted = lambda t, args_i: H(t+args_i["_t0"], args_i)
        else:
            raise TypeError("If using function-callback based Hamiltonian" +
                            "time-dependence, args must be a dictionary")
        return H_shifted, _args

    if isinstance(H, list):
        # string/function-list based time-dependence
        H_shifted = []
        if args is None:
            _args = {"_t0": 0}
        elif isinstance(args, dict):
            _args = args.copy()
            _args["_t0"] = 0
        else:
            _args = {"_user_args": args, "_t0": 0}

        for i in range(len(H)):
            if isinstance(H[i], list):
                # modify hamiltonian time dependence in accordance with the
                # quantum regression theorem
                if isinstance(args, dict) or args is None:
                    if isinstance(H[i][1], types.FunctionType):
                        # function-list based time-dependence
                        fn = lambda t, args_i: \
                            H[i][1](t+args_i["_t0"], args_i)
                    else:
                        # string-list based time-dependence
                        # Note: other functions already raise errors for mixed
                        # td formatting
                        fn = sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                                 "(t+_t0)", H[i][1])
                else:
                    if isinstance(H[i][1], types.FunctionType):
                        # function-list based time-dependence
                        fn = lambda t, args_i: \
                            H[i][1](t+args_i["_t0"], args_i["_user_args"])
                    else:
                        raise TypeError("If using string-list based" +
                                        "Hamiltonian time-dependence, args" +
                                        "must be a dictionary")
                H_shifted.append([H[i][0], fn])
            else:
                H_shifted.append(H[i])

        return H_shifted, _args
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

__all__ = [
    'correlation_2op_1t', 'correlation_2op_2t', 'correlation_3op_1t',
    'correlation_3op_2t', 'coherence_function_g1', 'coherence_function_g2',
    'spectrum', 'spectrum_correlation_fft',
]

from re import sub
import types

import numpy as np
import scipy.fftpack

from ..core import (
    qeye, Qobj, QobjEvo, liouvillian, spre, unstack_columns, stack_columns,
    tensor, qzero, expect
)
from .mesolve import mesolve
from .mcsolve import mcsolve
from ._rhs_generate import rhs_clear, td_wrap_array_str
from ._utilities import cython_build_cleanup
from .solver import SolverOptions, config
from .steadystate import steadystate
from ..settings import settings
debug = settings.core['debug']

if debug:
    import inspect


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

# low level correlation

def correlation_2op_1t(H, state0, taulist, c_ops, a_op, b_op,
                       solver="me", reverse=False, args={},
                       options=SolverOptions(ntraj=[20, 100])):
    r"""
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\tau)B(t)\right>`
    along one time axis using the quantum regression theorem and the evolution
    solver indicated by the `solver` parameter.

    Parameters
    ----------

    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    reverse : bool {False, True}
        If `True`, calculate :math:`\left<A(t)B(t+\tau)\right>` instead of
        :math:`\left<A(t+\tau)B(t)\right>`.
    solver : str {'me', 'mc', 'es'}
        choice of solver (`me` for master-equation, `mc` for Monte Carlo, and
        `es` for exponential series).
    options : SolverOptions
        Solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------
    corr_vec : ndarray
        An array of correlation values for the times specified by `tlist`.

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.

    """

    if debug:
        print(inspect.stack()[0][3])
    if reverse:
        A_op = a_op
        B_op = b_op
        C_op = 1
    else:
        A_op = 1
        B_op = a_op
        C_op = b_op
    return _correlation_2t(H, state0, [0], taulist, c_ops, A_op, B_op, C_op,
                           solver=solver, args=args, options=options)[0]


def correlation_2op_2t(H, state0, tlist, taulist, c_ops, a_op, b_op,
                       solver="me", reverse=False, args={},
                       options=SolverOptions(ntraj=[20, 100])):
    r"""
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\tau)B(t)\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\rho_0` or state vector
        :math:`\psi_0`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    tlist : array_like
        list of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \rightarrow \infty`; here
        tlist is automatically set, ignoring user input.
    taulist : array_like
        list of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    reverse : bool {False, True}
        If `True`, calculate :math:`\left<A(t)B(t+\tau)\right>` instead of
        :math:`\left<A(t+\tau)B(t)\right>`.
    solver : str
        choice of solver (`me` for master-equation, `mc` for Monte Carlo, and
        `es` for exponential series).
    options : SolverOptions
        solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------
    corr_mat : ndarray
        An 2-dimensional array (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional array of correlation values
        is returned instead.

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.

    """
    if debug:
        print(inspect.stack()[0][3])
    if tlist is None:
        return correlation_2op_1t(H, state0, taulist, c_ops, a_op, b_op,
                                  solver=solver, reverse=reverse, args=args,
                                  options=options)
    if reverse:
        A_op = a_op
        B_op = b_op
        C_op = 1
    else:
        A_op = 1
        B_op = a_op
        C_op = b_op
    return _correlation_2t(H, state0, tlist, taulist, c_ops, A_op, B_op, C_op,
                           solver=solver, args=args, options=options)


def correlation_3op_1t(H, state0, taulist, c_ops, a_op, b_op, c_op,
                       solver="me", args={},
                       options=SolverOptions(ntraj=[20, 100])):
    r"""
    Calculate the three-operator two-time correlation function:
    :math:`\left<A(t)B(t+\tau)C(t)\right>` along one time axis using the
    quantum regression theorem and the evolution solver indicated by the
    `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\tau<0`.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    rho0 : Qobj
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    c_op : Qobj
        operator C.
    solver : str
        choice of solver (`me` for master-equation, `mc` for Monte Carlo, and
        `es` for exponential series).
    options : Options
        solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------
    corr_vec : array
        An array of correlation values for the times specified by `taulist`

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.

    """
    if debug:
        print(inspect.stack()[0][3])
    return _correlation_2t(H, state0, [0], taulist, c_ops, a_op, b_op, c_op,
                           solver=solver, args=args, options=options)[0]


def correlation_3op_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                       solver="me", args={},
                       options=SolverOptions(ntraj=[20, 100])):
    r"""
    Calculate the three-operator two-time correlation function:
    :math:`\left<A(t)B(t+\tau)C(t)\right>` along two time axes using the
    quantum regression theorem and the evolution solver indicated by the
    `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\tau<0`.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    rho0 : Qobj
        Initial state density matrix :math:`\rho_0` or state vector
        :math:`\psi_0`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    tlist : array_like
        list of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \rightarrow \infty`; here
        tlist is automatically set, ignoring user input.
    taulist : array_like
        list of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    c_op : Qobj
        operator C.
    solver : str
        choice of solver (`me` for master-equation, `mc` for Monte Carlo, and
        `es` for exponential series).
    options : SolverOptions
        solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------
    corr_mat : array
        An 2-dimensional array (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional array of correlation values
        is returned instead.

    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.

    """
    if debug:
        print(inspect.stack()[0][3])
    if tlist is None:
        return correlation_3op_1t(H, state0, taulist, c_ops, a_op, b_op, c_op,
                                  solver=solver, args=args, options=options)
    return _correlation_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                           solver=solver, args=args, options=options)


# high level correlation

def coherence_function_g1(H, state0, taulist, c_ops, a_op, solver="me",
                          args={}, options=SolverOptions(ntraj=[20, 100])):
    r"""
    Calculate the normalized first-order quantum coherence function:

    .. math::

        g^{(1)}(\tau) =
        \frac{\langle A^\dagger(\tau)A(0)\rangle}
        {\sqrt{\langle A^\dagger(\tau)A(\tau)\rangle
                \langle A^\dagger(0)A(0)\rangle}}

    using the quantum regression theorem and the evolution solver indicated by
    the `solver` parameter.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series).
    options : SolverOptions
        solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------
    g1, G1 : tuple
        The normalized and unnormalized second-order coherence function.

    """
    # first calculate the photon number
    if state0 is None:
        state0 = steadystate(H, c_ops)
        n = np.array([expect(state0, a_op.dag() * a_op)])
    else:
        n = mesolve(H, state0, taulist, c_ops, [a_op.dag() * a_op],
                    options=options).expect[0]

    # calculate the correlation function G1 and normalize with n to obtain g1
    G1 = correlation_2op_1t(H, state0, taulist, c_ops, a_op.dag(), a_op,
                            solver=solver, args=args, options=options)
    g1 = G1 / np.sqrt(n[0] * n)
    return g1, G1


def coherence_function_g2(H, state0, taulist, c_ops, a_op, solver="me",
                          args={}, options=SolverOptions(ntraj=[20, 100])):
    r"""
    Calculate the normalized second-order quantum coherence function:

    .. math::

         g^{(2)}(\tau) =
        \frac{\langle A^\dagger(0)A^\dagger(\tau)A(\tau)A(0)\rangle}
        {\langle A^\dagger(\tau)A(\tau)\rangle
         \langle A^\dagger(0)A(0)\rangle}

    using the quantum regression theorem and the evolution solver indicated by
    the `solver` parameter.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    args : dict
        Dictionary of arguments to be passed to solver.
    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series).
    options : SolverOptions
        solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------
    g2, G2 : tuple
        The normalized and unnormalized second-order coherence function.

    """
    # first calculate the photon number
    if state0 is None:
        state0 = steadystate(H, c_ops)
        n = np.array([expect(state0, a_op.dag() * a_op)])
    else:
        n = mesolve(H, state0, taulist, c_ops, [a_op.dag() * a_op], args=args).expect[0]

    # calculate the correlation function G2 and normalize with n to obtain g2
    G2 = correlation_3op_1t(H, state0, taulist, c_ops,
                            a_op.dag(), a_op.dag()*a_op, a_op,
                            solver=solver, args=args, options=options)
    g2 = G2 / (n[0] * n)
    return g2, G2


# spectrum

def spectrum(H, wlist, c_ops, a_op, b_op, solver="es", use_pinv=False):
    r"""
    Calculate the spectrum of the correlation function
    :math:`\lim_{t \to \infty} \left<A(t+\tau)B(t)\right>`,
    i.e., the Fourier transform of the correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty}
        \lim_{t \to \infty} \left<A(t+\tau)B(t)\right>
        e^{-i\omega\tau} d\tau.

    using the solver indicated by the `solver` parameter. Note: this spectrum
    is only defined for stationary statistics (uses steady state rho0)

    Parameters
    ----------
    H : :class:`qutip.qobj`
        system Hamiltonian.
    wlist : array_like
        list of frequencies for :math:`\omega`.
    c_ops : list
        list of collapse operators.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    solver : str
        choice of solver (`es` for exponential series and
        `pi` for psuedo-inverse).
    use_pinv : bool
        For use with the `pi` solver: if `True` use numpy's pinv method,
        otherwise use a generic solver.

    Returns
    -------
    spectrum : array
        An array with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """
    if debug:
        print(inspect.stack()[0][3])
    if solver == "es":
        return _spectrum_es(H, wlist, c_ops, a_op, b_op)
    elif solver == "pi":
        return _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv)
    raise ValueError("Unrecognized choice of solver {} (use es or pi)."
                     .format(solver))


def spectrum_correlation_fft(tlist, y, inverse=False):
    """
    Calculate the power spectrum corresponding to a two-time correlation
    function using FFT.

    Parameters
    ----------
    tlist : array_like
        list/array of times :math:`t` which the correlation function is given.
    y : array_like
        list/array of correlations corresponding to time delays :math:`t`.
    inverse: boolean
        boolean parameter for using a positive exponent in the Fourier Transform instead. Default is False.

    Returns
    -------
    w, S : tuple
        Returns an array of angular frequencies 'w' and the corresponding
        two-sided power spectrum 'S(w)'.

    """
    if debug:
        print(inspect.stack()[0][3])
    tlist = np.asarray(tlist)
    N = tlist.shape[0]
    dt = tlist[1] - tlist[0]
    if not np.allclose(np.diff(tlist), dt*np.ones(N-1, dtype=float)):
        raise ValueError('tlist must be equally spaced for FFT.')
    F = (N*scipy.fftpack.ifft(y)) if inverse else scipy.fftpack.fft(y)
    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)
    # re-order frequencies from most negative to most positive (centre on 0)
    idx = np.array([], dtype='int')
    idx = np.append(idx, np.where(f < 0.0))
    idx = np.append(idx, np.where(f >= 0.0))
    return 2*np.pi*f[idx], 2*dt*np.real(F[idx])


# -----------------------------------------------------------------------------
# PRIVATE SOLVER METHODS
# -----------------------------------------------------------------------------

# master 2t correlation solver

def _correlation_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                    solver="me", args={}, options=SolverOptions()):
    """
    Internal function for calling solvers in order to calculate the
    three-operator two-time correlation function:
    <A(t)B(t+tau)C(t)>
    """

    # Note: the current form of the correlator is sufficient for all possible
    # two-time correlations (incuding those with 2ops vs 3). Ex: to compute a
    # correlation of the form <A(t+tau)B(t)>: a_op = identity, b_op = A,
    # and c_op = B.

    if debug:
        print(inspect.stack()[0][3])

    if min(tlist) != 0:
        raise TypeError("tlist must be positive and contain the element 0.")
    if min(taulist) != 0:
        raise TypeError("taulist must be positive and contain the element 0.")

    if config.tdname:
        cython_build_cleanup(config.tdname)
    rhs_clear()
    H, c_ops, args = td_wrap_array_str(H, c_ops, args, tlist)

    if solver == "me":
        return _correlation_me_2t(H, state0, tlist, taulist,
                                  c_ops, a_op, b_op, c_op,
                                  args=args, options=options)
    elif solver == "mc":
        return _correlation_mc_2t(H, state0, tlist, taulist,
                                  c_ops, a_op, b_op, c_op,
                                  args=args, options=options)
    elif solver == "es":
        return _correlation_es_2t(H, state0, tlist, taulist,
                                  c_ops, a_op, b_op, c_op)
    raise ValueError("Unrecognized choice of solver" +
                     "%s (use me, mc, or es)." % solver)


# master equation solvers

def _correlation_me_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                       args={}, options=SolverOptions()):
    """
    Internal function for calculating the three-operator two-time
    correlation function:
    <A(t)B(t+tau)C(t)>
    using a master equation solver.
    """

    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if state0 is None:
        rho0 = steadystate(H, c_ops)
        tlist = [0]
    elif state0.isket:
        rho0 = state0.proj()
    else:
        rho0 = state0

    if debug:
        print(inspect.stack()[0][3])

    rho_t = mesolve(H, rho0, tlist, c_ops, args=args, options=options).states
    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    H = QobjEvo(H, args=args, tlist=tlist, copy=False)
    c_ops = [QobjEvo(op, args=args, tlist=tlist, copy=False) for op in c_ops]

    for t_idx, rho in enumerate(rho_t):
        H_shifted = H._insert_time_shift(tlist[t_idx])
        c_ops_shifted = [op._insert_time_shift(tlist[t_idx]) for op in c_ops]

        corr_mat[t_idx, :] = mesolve(
            H_shifted, c_op * rho * a_op, taulist, c_ops_shifted,
            [b_op], args=args, options=options
        ).expect[0]

    return corr_mat


# exponential series solvers

def _correlation_es_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op):
    """
    Internal function for calculating the three-operator two-time
    correlation function:
    <A(t)B(t+tau)C(t)>
    using an exponential series solver.
    """

    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if state0 is None:
        rho0 = steadystate(H, c_ops)
        tlist = [0]
    elif state0.isket:
        rho0 = state0.proj()
    else:
        rho0 = state0

    if debug:
        print(inspect.stack()[0][3])

    # contruct the Liouvillian
    L = liouvillian(H, c_ops)
    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    solES_t = QobjEvo([[rho, _exponential_term(w)]
                       for rho, w in zip(*_diagonal_evolution(L, rho0))])
    rho_t = c_op * solES_t * a_op
    # evaluate the correlation function
    for i, t in enumerate(tlist):
        states, rates = _diagonal_evolution(L, rho_t(t))
        expects = np.array([expect(b_op, state) for state in states])
        for k, tau in enumerate(taulist):
            corr_mat[i, k] = expects @ np.exp(rates * tau)
    return corr_mat


def _spectrum_es(H, wlist, c_ops, a_op, b_op):
    r"""
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\tau)B(0)\right>`.
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
    states, rates = _diagonal_evolution(L, b_op * rho0)
    # correlation
    ampls = expect(a_op, states)
    # make covariance
    ampls = np.concatenate([ampls, [-a_op_ss*b_op_ss]])
    rates = np.concatenate([rates, [0]])
    # Tidy up similar rates.
    uniques = {}
    for r, a in zip(rates, ampls):
        for r_ in uniques:
            if np.abs(r - r_) < 1e-10:
                uniques[r_] += a
                break
        else:
            uniques[r] = a
    ampls, rates = [], []
    for r, a in uniques.items():
        if np.abs(a) > 1e-10:
            ampls.append(a)
            rates.append(r)
    ampls, rates = np.array(ampls), np.array(rates)
    return np.array([2*np.dot(ampls, 1/(1j * w - rates)).real for w in wlist])


# Monte Carlo solvers

def _correlation_mc_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                       args={}, options=SolverOptions()):
    """
    Internal function for calculating the three-operator two-time
    correlation function:
    <A(t)B(t+tau)C(t)>
    using a Monte Carlo solver.
    """
    if not c_ops:
        raise TypeError("If no collapse operators are required, use the `me`" +
                        "or `es` solvers")
    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if state0 is None:
        raise NotImplementedError("steady state not implemented for " +
                                  "mc solver, please use `es` or `me`")
    if not state0.isket:
        raise TypeError("state0 must be a state vector.")
    psi0 = state0

    if debug:
        print(inspect.stack()[0][3])

    psi_t_mat = mcsolve(
        H, psi0, tlist, c_ops, [],
        args=args, ntraj=options['ntraj'][0], options=options,
        progress_bar=None,
    ).states

    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    H = QobjEvo(H, args=args, tlist=tlist, copy=False)
    c_ops = [QobjEvo(op, args=args, tlist=tlist, copy=False) for op in c_ops]

    # calculation of <A(t)B(t+tau)C(t)> from only knowledge of psi0 requires
    # averaging over both t and tau
    for t_idx in range(np.size(tlist)):
        H_shifted = H._insert_time_shift(tlist[t_idx])
        c_ops_shifted = [op._insert_time_shift(tlist[t_idx]) for op in c_ops]

        for trial_idx in range(options['ntraj'][0]):
            if isinstance(a_op, Qobj) and isinstance(c_op, Qobj):
                if a_op.dag() == c_op:
                    # A shortcut here, requires only 1/4 the trials
                    chi_0 = (options['mc_corr_eps'] + c_op) * \
                        psi_t_mat[trial_idx, t_idx]

                    # evolve these states and calculate expectation value of B
                    c_tau = chi_0.norm()**2 * mcsolve(
                        H_shifted, chi_0/chi_0.norm(), taulist, c_ops_shifted,
                        [b_op],
                        args=args, ntraj=options['ntraj'][1], options=options,
                        progress_bar=None
                    ).expect[0]

                    # final correlation vector computed by combining the
                    # averages
                    corr_mat[t_idx, :] += c_tau/options['ntraj'][1]
            else:
                # otherwise, need four trial wavefunctions
                # (Ad+C)*psi_t, (Ad+iC)*psi_t, (Ad-C)*psi_t, (Ad-iC)*psi_t
                if isinstance(a_op, Qobj):
                    a_op_dag = a_op.dag()
                else:
                    # assume this is a number, ex. i.e. a_op = 1
                    # if this is not correct, the over-loaded addition
                    # operation will raise errors
                    a_op_dag = a_op
                chi_0 = [(options['mc_corr_eps'] + a_op_dag +
                          np.exp(1j*x*np.pi/2)*c_op) *
                         psi_t_mat[trial_idx, t_idx]
                         for x in range(4)]

                # evolve these states and calculate expectation value of B
                c_tau = [
                    chi.norm()**2 * mcsolve(
                        H_shifted, chi/chi.norm(), taulist, c_ops_shifted,
                        [b_op],
                        args=args, ntraj=options['ntraj'][1], options=options,
                        progress_bar=None
                    ).expect[0]
                    for chi in chi_0
                ]

                # final correlation vector computed by combining the averages
                corr_mat_add = np.asarray(
                    1.0 / (4*options['ntraj'][0]) *
                    (c_tau[0] - c_tau[2] - 1j*c_tau[1] + 1j*c_tau[3]),
                    dtype=corr_mat.dtype
                )
                corr_mat[t_idx, :] += corr_mat_add

    return corr_mat


# pseudo-inverse solvers
def _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv=False):
    r"""
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\tau)B(0)\right>`.
    """
    L = H if H.issuper else liouvillian(H, c_ops)
    tr_mat = tensor([qeye(n) for n in L.dims[0][0]])
    N = np.prod(L.dims[0][0])
    A = L.full()
    b = spre(b_op).full()
    a = spre(a_op).full()

    tr_vec = np.transpose(stack_columns(tr_mat.full()))

    rho_ss = steadystate(L)
    rho = np.transpose(stack_columns(rho_ss.full()))

    I = np.identity(N * N)
    P = np.kron(np.transpose(rho), tr_vec)
    Q = I - P

    spectrum = np.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if use_pinv:
            MMR = np.linalg.pinv(-1.0j * w * I + A)
        else:
            MMR = np.dot(Q, np.linalg.solve(-1.0j * w * I + A, Q))

        s = np.dot(tr_vec,
                   np.dot(a, np.dot(MMR, np.dot(b, np.transpose(rho)))))
        spectrum[idx] = -2 * np.real(s[0, 0])
    return spectrum


class _exponential_term:
    def __init__(self, rate, amplitude=1):
        self.amplitude = amplitude
        self.rate = rate

    def __call__(self, t, args=None):
        return self.amplitude * np.exp(self.rate * t)


def _diagonal_evolution(L, rho0):
    """
    Diagonalise the evolution of density matrix rho0 under a constant
    Liouvillian L.  Returns a list of `states` and an array of the eigenvalues
    such that the time evolution of rho0 is represented by
        sum_k states[k] * exp(evals[k] * t)
    This is effectively the same as the legacy QuTiP function ode2es, but does
    not use the removed eseries class.  It exists here because ode2es and
    essolve were removed.
    """
    rho0_full = rho0.full()
    if np.abs(rho0_full).sum() < 1e-10 + 1e-24:
        return qzero(rho0.dims[0]), np.array([0])
    evals, evecs = L.eigenstates()
    evecs = np.vstack([ket.full()[:, 0] for ket in evecs]).T
    # evals[i]   = eigenvalue i
    # evecs[:, i] = eigenvector i
    size = rho0.shape[0] * rho0.shape[1]
    r0 = stack_columns(rho0_full)[:, 0]
    v0 = scipy.linalg.solve(evecs, r0)
    vv = evecs * v0[None, :]  # product equivalent to `evecs @ np.diag(v0)`
    states = [Qobj(unstack_columns(vv[:, i]), dims=rho0.dims, type='oper')
              for i in range(size)]
    # We don't use QobjEvo because it isn't designed to be efficient when
    # calculating
    return states, evals

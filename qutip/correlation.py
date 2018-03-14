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

__all__ = ['correlation_2op_1t', 'correlation_2op_2t', 'correlation_3op_1t',
           'correlation_3op_2t', 'coherence_function_g1',
           'coherence_function_g2', 'spectrum', 'spectrum_correlation_fft',
           'correlation_ss', 'correlation', 'correlation_4op_1t',
           'correlation_4op_2t', 'spectrum_ss', 'spectrum_pi']

from re import sub
from warnings import warn
import types

import numpy as np
import scipy.fftpack

from qutip.eseries import esval, esspec
from qutip.essolve import ode2es
from qutip.expect import expect
from qutip.mesolve import mesolve
from qutip.mcsolve import mcsolve
from qutip.operators import qeye
from qutip.qobj import Qobj, isket, issuper
from qutip.rhs_generate import rhs_clear, _td_wrap_array_str
from qutip.cy.utilities import _cython_build_cleanup
from qutip.settings import debug
from qutip.solver import Options, config
from qutip.steadystate import steadystate
from qutip.states import ket2dm
from qutip.superoperator import liouvillian, spre, mat2vec
from qutip.tensor import tensor

if debug:
    import inspect


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

# low level correlation

def correlation_2op_1t(H, state0, taulist, c_ops, a_op, b_op,
                       solver="me", reverse=False, args={},
                       options=Options(ntraj=[20, 100])):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\\tau)B(t)\\right>`
    along one time axis using the quantum regression theorem and the evolution
    solver indicated by the `solver` parameter.

    Parameters
    ----------

    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    reverse : bool {False, True}
        If `True`, calculate :math:`\left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\left<A(t+\\tau)B(t)\\right>`.
    solver : str {'me', 'mc', 'es'}
        choice of solver (`me` for master-equation, `mc` for Monte Carlo, and
        `es` for exponential series).
    options : Options
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
                       options=Options(ntraj=[20, 100])):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\\tau)B(t)\\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\\rho_0` or state vector
        :math:`\\psi_0`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    tlist : array_like
        list of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \\rightarrow \\infty`; here
        tlist is automatically set, ignoring user input.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    b_op : Qobj
        operator B.
    reverse : bool {False, True}
        If `True`, calculate :math:`\left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\left<A(t+\\tau)B(t)\\right>`.
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
    else:
        if reverse:
            A_op = a_op
            B_op = b_op
            C_op = 1
        else:
            A_op = 1
            B_op = a_op
            C_op = b_op

        return _correlation_2t(H, state0, tlist, taulist,
                               c_ops, A_op, B_op, C_op,
                               solver=solver, args=args, options=options)


def correlation_3op_1t(H, state0, taulist, c_ops, a_op, b_op, c_op,
                       solver="me", args={},
                       options=Options(ntraj=[20, 100])):
    """
    Calculate the three-operator two-time correlation function:
    :math:`\left<A(t)B(t+\\tau)C(t)\\right>`
    along one time axis using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\\tau<0`.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    rho0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
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
                       options=Options(ntraj=[20, 100])):
    """
    Calculate the three-operator two-time correlation function:
    :math:`\left<A(t)B(t+\\tau)C(t)\\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\\tau<0`.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    rho0 : Qobj
        Initial state density matrix :math:`\\rho_0` or state vector
        :math:`\\psi_0`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    tlist : array_like
        list of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \\rightarrow \\infty`; here
        tlist is automatically set, ignoring user input.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
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
    else:
        return _correlation_2t(H, state0, tlist, taulist,
                               c_ops, a_op, b_op, c_op,
                               solver=solver, args=args, options=options)


# high level correlation

def coherence_function_g1(H, state0, taulist, c_ops, a_op, solver="me",
                          args={}, options=Options(ntraj=[20, 100])):
    """
    Calculate the normalized first-order quantum coherence function:

    .. math::

        g^{(1)}(\\tau) =
        \\frac{\\langle A^\\dagger(\\tau)A(0)\\rangle}
        {\\sqrt{\\langle A^\\dagger(\\tau)A(\\tau)\\rangle
                \\langle A^\\dagger(0)A(0)\\rangle}}

    using the quantum regression theorem and the evolution solver indicated by
    the `solver` parameter.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.
    a_op : Qobj
        operator A.
    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series).
    options : Options
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


def coherence_function_g2(H, state0, taulist, c_ops, a_op, solver="me", args={},
                          options=Options(ntraj=[20, 100])):
    """
    Calculate the normalized second-order quantum coherence function:

    .. math::

         g^{(2)}(\\tau) =
        \\frac{\\langle A^\\dagger(0)A^\\dagger(\\tau)A(\\tau)A(0)\\rangle}
        {\\langle A^\\dagger(\\tau)A(\\tau)\\rangle
         \\langle A^\\dagger(0)A(0)\\rangle}

    using the quantum regression theorem and the evolution solver indicated by
    the `solver` parameter.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    state0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
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
    options : Options
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
    """
    Calculate the spectrum of the correlation function
    :math:`\lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>`,
    i.e., the Fourier transform of the correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty}
        \lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>
        e^{-i\omega\\tau} d\\tau.

    using the solver indicated by the `solver` parameter. Note: this spectrum
    is only defined for stationary statistics (uses steady state rho0)

    Parameters
    ----------
    H : :class:`qutip.qobj`
        system Hamiltonian.
    wlist : array_like
        list of frequencies for :math:`\\omega`.
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
    else:
        raise ValueError("Unrecognized choice of solver" +
                         "%s (use es or pi)." % solver)


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
    if not np.allclose(np.diff(tlist), dt*np.ones(N-1,dtype=float)):
        raise Exception('tlist must be equally spaced for FFT.')
    
    if inverse:
           F = N * scipy.fftpack.ifft(y)
    else:
           F = scipy.fftpack.fft(y)
    
    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)

    # re-order frequencies from most negative to most positive (centre on 0)
    idx = np.array([], dtype = 'int')
    idx = np.append(idx, np.where(f < 0.0))
    idx = np.append(idx, np.where(f >= 0.0))

    return 2 * np.pi * f[idx], 2 * dt * np.real(F[idx])


# -----------------------------------------------------------------------------
# LEGACY API
# -----------------------------------------------------------------------------

# low level correlation

def correlation_ss(H, taulist, c_ops, a_op, b_op,
                   solver="me", reverse=False, args={},
                   options=Options(ntraj=[20, 100])):
    """
    Calculate the two-operator two-time correlation function:

    .. math::

        \lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>

    along one time axis (given steady-state initial conditions) using the
    quantum regression theorem and the evolution solver indicated by the
    `solver` parameter.

    Parameters
    ----------

    H : Qobj
        system Hamiltonian.

    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
        the element `0`.

    c_ops : list
        list of collapse operators.

    a_op : Qobj
        operator A.

    b_op : Qobj
        operator B.

    reverse : *bool*
        If `True`, calculate
        :math:`\lim_{t \\to \\infty} \left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>`.

    solver : str
        choice of solver (`me` for master-equation and
        `es` for exponential series).

    options : Options
        solver options class. `ntraj` is taken as a two-element list because
        the `mc` correlator calls `mcsolve()` recursively; by default,
        `ntraj=[20, 100]`. `mc_corr_eps` prevents divide-by-zero errors in
        the `mc` correlator; by default, `mc_corr_eps=1e-10`.

    Returns
    -------

    corr_vec : array
        An array of correlation values for the times specified by `tlist`.

    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.

    """

    warn("correlation_ss() now legacy, please use correlation_2op_1t() with" +
         "initial state as None", FutureWarning)

    if debug:
        print(inspect.stack()[0][3])

    return correlation_2op_1t(H, None, taulist, c_ops, a_op, b_op,
                              solver=solver, reverse=reverse, args=args,
                              options=options)


def correlation(H, state0, tlist, taulist, c_ops, a_op, b_op,
                solver="me", reverse=False, args={},
                options=Options(ntraj=[20, 100])):
    """
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\\tau)B(t)\\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Parameters
    ----------

    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.

    state0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.

    tlist : array_like
        list of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \\rightarrow \\infty`; here
        tlist is automatically set, ignoring user input.

    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
        the element `0`.

    c_ops : list
        list of collapse operators, may be time-dependent for solver choice of
        `me` or `mc`.

    a_op : Qobj
        operator A.

    b_op : Qobj
        operator B.

    reverse : *bool*
        If `True`, calculate :math:`\left<A(t)B(t+\\tau)\\right>` instead of
        :math:`\left<A(t+\\tau)B(t)\\right>`.

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

    corr_mat : array
        An 2-dimensional array (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional array of correlation values
        is returned instead.

    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.

    """

    warn("correlation() now legacy, please use correlation_2op_2t()",
         FutureWarning)

    if debug:
        print(inspect.stack()[0][3])

    return correlation_2op_2t(H, state0, tlist, taulist, c_ops, a_op, b_op,
                              solver=solver, reverse=reverse, args=args,
                              options=options)


def correlation_4op_1t(H, state0, taulist, c_ops, a_op, b_op, c_op, d_op,
                       solver="me", args={},
                       options=Options(ntraj=[20, 100])):
    """
    Calculate the four-operator two-time correlation function:
    :math:`\left<A(t)B(t+\\tau)C(t+\\tau)D(t)\\right>`
    along one time axis using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\\tau<0`.

    Parameters
    ----------
    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.
    rho0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.
    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
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

    d_op : Qobj
        operator D.

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
        An array of correlation values for the times specified by `taulist`.

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.
                       
    .. note:: Deprecated in QuTiP 3.1
              Use correlation_3op_1t() instead.

    """

    warn("correlation_4op_1t() now legacy, please use correlation_3op_1t()",
         FutureWarning)
    warn("the reverse argument has been removed as it did not contain any" +
         "new physical information", DeprecationWarning)

    if debug:
        print(inspect.stack()[0][3])

    return correlation_3op_1t(H, state0, taulist, c_ops,
                              a_op, b_op * c_op, d_op,
                              solver=solver, args=args, options=options)


def correlation_4op_2t(H, state0, tlist, taulist, c_ops,
                       a_op, b_op, c_op, d_op, solver="me", args={},
                       options=Options(ntraj=[20, 100])):
    """
    Calculate the four-operator two-time correlation function:
    :math:`\left<A(t)B(t+\\tau)C(t+\\tau)D(t)\\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\\tau<0`.

    Parameters
    ----------

    H : Qobj
        system Hamiltonian, may be time-dependent for solver choice of `me` or
        `mc`.

    rho0 : Qobj
        Initial state density matrix :math:`\\rho_0` or state vector
        :math:`\\psi_0`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        for the `me` and `es` solvers.

    tlist : array_like
        list of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \\rightarrow \\infty`; here
        tlist is automatically set, ignoring user input.

    taulist : array_like
        list of times for :math:`\\tau`. taulist must be positive and contain
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

    d_op : Qobj
        operator D.

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

    corr_mat : array
        An 2-dimensional array (matrix) of correlation values for the times
        specified by `tlist` (first index) and `taulist` (second index). If
        `tlist` is `None`, then a 1-dimensional array of correlation values
        is returned instead.

    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.

    """

    warn("correlation_4op_2t() now legacy, please use correlation_3op_2t()",
         FutureWarning)
    warn("the reverse argument has been removed as it did not contain any" +
         "new physical information", DeprecationWarning)

    if debug:
        print(inspect.stack()[0][3])

    return correlation_3op_2t(H, state0, tlist, taulist, c_ops,
                              a_op, b_op * c_op, d_op,
                              solver=solver, args=args, options=options)


# spectrum

def spectrum_ss(H, wlist, c_ops, a_op, b_op):
    """
    Calculate the spectrum of the correlation function
    :math:`\lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>`,
    i.e., the Fourier transform of the correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty}
        \lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>
        e^{-i\omega\\tau} d\\tau.

    using an eseries based solver Note: this spectrum is only defined for
    stationary statistics (uses steady state rho0).

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    wlist : array_like
        list of frequencies for :math:`\\omega`.

    c_ops : *list* of :class:`qutip.qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj`
        operator A.

    b_op : :class:`qutip.qobj`
        operator B.

    use_pinv : *bool*
        If `True` use numpy's `pinv` method, otherwise use a generic solver.

    Returns
    -------

    spectrum : array
        An array with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """

    warn("spectrum_ss() now legacy, please use spectrum()", FutureWarning)

    return spectrum(H, wlist, c_ops, a_op, b_op, solver="es")


def spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv=False):
    """
    Calculate the spectrum of the correlation function
    :math:`\lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>`,
    i.e., the Fourier transform of the correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty}
        \lim_{t \\to \\infty} \left<A(t+\\tau)B(t)\\right>
        e^{-i\omega\\tau} d\\tau.

    using a psuedo-inverse method. Note: this spectrum is only defined for
    stationary statistics (uses steady state rho0)

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    wlist : array_like
        list of frequencies for :math:`\\omega`.

    c_ops : *list* of :class:`qutip.qobj`
        list of collapse operators.

    a_op : :class:`qutip.qobj`
        operator A.

    b_op : :class:`qutip.qobj`
        operator B.

    use_pinv : *bool*
        If `True` use numpy's pinv method, otherwise use a generic solver.

    Returns
    -------

    spectrum : array
        An array with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """

    warn("spectrum_pi() now legacy, please use spectrum()", FutureWarning)

    return spectrum(H, wlist, c_ops, a_op, b_op,
                    solver="pi", use_pinv=use_pinv)


# -----------------------------------------------------------------------------
# PRIVATE SOLVER METHODS
# -----------------------------------------------------------------------------

# master 2t correlation solver

def _correlation_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                    solver="me", args={}, options=Options()):
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
        _cython_build_cleanup(config.tdname)
    rhs_clear()
    H, c_ops, args = _td_wrap_array_str(H, c_ops, args, tlist)

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
    else:
        raise ValueError("Unrecognized choice of solver" +
                         "%s (use me, mc, or es)." % solver)


# master equation solvers

def _correlation_me_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                       args={}, options=Options()):
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
    elif isket(state0):
        rho0 = ket2dm(state0)
    else:
        rho0 = state0

    if debug:
        print(inspect.stack()[0][3])

    rho_t = mesolve(H, rho0, tlist, c_ops, [],
                    args=args, options=options).states
    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    H_shifted, c_ops_shifted, _args = _transform_L_t_shift(H, c_ops, args)
    if config.tdname:
        _cython_build_cleanup(config.tdname)
    rhs_clear()

    for t_idx, rho in enumerate(rho_t):
        if not isinstance(H, Qobj):
            _args["_t0"] = tlist[t_idx]

        corr_mat[t_idx, :] = mesolve(
            H_shifted, c_op * rho * a_op, taulist, c_ops_shifted,
            [b_op], args=_args, options=options
        ).expect[0]

        if t_idx == 1:
            options.rhs_reuse = True

    if config.tdname:
        _cython_build_cleanup(config.tdname)
    rhs_clear()

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
    elif isket(state0):
        rho0 = ket2dm(state0)
    else:
        rho0 = state0

    if debug:
        print(inspect.stack()[0][3])

    # contruct the Liouvillian
    L = liouvillian(H, c_ops)

    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    solES_t = ode2es(L, rho0)

    # evaluate the correlation function
    for t_idx in range(len(tlist)):
        rho_t = esval(solES_t, [tlist[t_idx]])
        solES_tau = ode2es(L, c_op * rho_t * a_op)
        corr_mat[t_idx, :] = esval(expect(b_op, solES_tau), taulist)

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
    cov_es = corr_es - a_op_ss * b_op_ss
    # tidy up covariance (to combine, e.g., zero-frequency components that cancel)
    cov_es.tidyup()

    # spectrum
    spectrum = esspec(cov_es, wlist)

    return spectrum


# Monte Carlo solvers

def _correlation_mc_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                       args={}, options=Options()):
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
    elif not isket(state0):
        raise TypeError("state0 must be a state vector.")
    psi0 = state0

    if debug:
        print(inspect.stack()[0][3])

    psi_t_mat = mcsolve(
        H, psi0, tlist, c_ops, [],
        args=args, ntraj=options.ntraj[0], options=options, progress_bar=None
    ).states

    corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
    H_shifted, c_ops_shifted, _args = _transform_L_t_shift(H, c_ops, args)
    if config.tdname:
        _cython_build_cleanup(config.tdname)
    rhs_clear()

    # calculation of <A(t)B(t+tau)C(t)> from only knowledge of psi0 requires
    # averaging over both t and tau
    for t_idx in range(np.size(tlist)):
        if not isinstance(H, Qobj):
            _args["_t0"] = tlist[t_idx]

        for trial_idx in range(options.ntraj[0]):
            if isinstance(a_op, Qobj) and isinstance(c_op, Qobj):
                if a_op.dag() == c_op:
                    # A shortcut here, requires only 1/4 the trials
                    chi_0 = (options.mc_corr_eps + c_op) * \
                        psi_t_mat[trial_idx, t_idx]

                    # evolve these states and calculate expectation value of B
                    c_tau = chi_0.norm()**2 * mcsolve(
                        H_shifted, chi_0/chi_0.norm(), taulist, c_ops_shifted,
                        [b_op],
                        args=_args, ntraj=options.ntraj[1], options=options,
                        progress_bar=None
                    ).expect[0]

                    # final correlation vector computed by combining the
                    # averages
                    corr_mat[t_idx, :] += c_tau/options.ntraj[1]
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
                chi_0 = [(options.mc_corr_eps + a_op_dag +
                          np.exp(1j*x*np.pi/2)*c_op) *
                         psi_t_mat[trial_idx, t_idx]
                         for x in range(4)]

                # evolve these states and calculate expectation value of B
                c_tau = [
                    chi.norm()**2 * mcsolve(
                        H_shifted, chi/chi.norm(), taulist, c_ops_shifted,
                        [b_op],
                        args=_args, ntraj=options.ntraj[1], options=options,
                        progress_bar=None
                    ).expect[0]
                    for chi in chi_0
                ]

                # final correlation vector computed by combining the averages
                corr_mat_add = np.asarray(
                    1.0 / (4*options.ntraj[0]) *
                    (c_tau[0] - c_tau[2] - 1j*c_tau[1] + 1j*c_tau[3]),
                    dtype=corr_mat.dtype
                )
                corr_mat[t_idx, :] += corr_mat_add
                    
        if t_idx == 1:
            options.rhs_reuse = True
    
    if config.tdname:
        _cython_build_cleanup(config.tdname)
    rhs_clear()

    return corr_mat


# pseudo-inverse solvers

def _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv=False):
    """
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\\tau)B(0)\\right>`.
    """

    L = H if issuper(H) else liouvillian(H, c_ops)

    tr_mat = tensor([qeye(n) for n in L.dims[0][0]])
    N = np.prod(L.dims[0][0])

    A = L.full()
    b = spre(b_op).full()
    a = spre(a_op).full()

    tr_vec = np.transpose(mat2vec(tr_mat.full()))

    rho_ss = steadystate(L)
    rho = np.transpose(mat2vec(rho_ss.full()))

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


# auxiliary

def _transform_L_t_shift(H, c_ops, args={}):
    """
    Time shift the Hamiltonian with private time-shift variable _t0
    """

    # while this list doesn't seem exhaustive, mesolve has already been called
    # and has hence called _td_format_check from qutip.rhs_generate. important
    # to keep in mind is that mesolve already requires the types of
    # time-dependence to be the same for the hamiltonian as for the collapse
    # operators.


    if isinstance(H, Qobj):
        # constant hamiltonian
        H_shifted = H  # not shifted!

    c_ops_is_td = False
    if isinstance(c_ops, list):
        for i in range(len(c_ops)):
            # test is collapse operators are time-dependent
            if isinstance(c_ops[i], list):
                c_ops_is_td = True

    if not c_ops_is_td:
        # constant collapse operators
        c_ops_shifted = c_ops  # not shifted!

    if isinstance(H, Qobj) and not c_ops_is_td:
        # constant hamiltonian and collapse operators
        _args = args  # not shifted!

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

    if isinstance(H, list) or c_ops_is_td:
        # string/function-list based time-dependence
        if args is None:
            _args = {"_t0": 0}
        elif isinstance(args, dict):
            _args = args.copy()
            _args["_t0"] = 0
        else:
            _args = {"_user_args": args, "_t0": 0}

        if isinstance(H, list):
            # hamiltonian is time-dependent
            H_shifted = []

            for i in range(len(H)):
                if isinstance(H[i], list):
                    # modify Hamiltonian time dependence in accordance with the
                    # quantum regression theorem
                    if isinstance(args, dict) or args is None:
                        if isinstance(H[i][1], types.FunctionType):
                            # function-list based time-dependence
                            fn = lambda t, args_i: \
                                H[i][1](t + args_i["_t0"], args_i)
                        else:
                            # string-list based time-dependence
                            # Again, note: _td_format_check already raises
                            # errors formixed td formatting
                            fn = sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                                     "(t+_t0)", H[i][1])
                    else:
                        if isinstance(H[i][1], types.FunctionType):
                            # function-list based time-dependence
                            fn = lambda t, args_i: \
                                H[i][1](t + args_i["_t0"],
                                        args_i["_user_args"])
                        else:
                            raise TypeError("If using string-list based" +
                                            "Hamiltonian time-dependence, " +
                                            "args must be a dictionary")
                    H_shifted.append([H[i][0], fn])
                else:
                    H_shifted.append(H[i])

        if c_ops_is_td:
            # collapse operators are time-dependent
            c_ops_shifted = []

            for i in range(len(c_ops)):
                if isinstance(c_ops[i], list):
                    # modify collapse operators time dependence in accordance
                    # with the quantum regression theorem
                    if isinstance(args, dict) or args is None:
                        if isinstance(c_ops[i][1], types.FunctionType):
                            # function-list based time-dependence
                            fn = lambda t, args_i: \
                                c_ops[i][1](t + args_i["_t0"], args_i)
                        else:
                            # string-list based time-dependence
                            # Again, note: _td_format_check already raises
                            # errors formixed td formatting
                            fn = sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                                     "(t+_t0)", c_ops[i][1])
                    else:
                        if isinstance(H[i][1], types.FunctionType):
                            # function-list based time-dependence
                            fn = lambda t, args_i: \
                                c_ops[i][1](t + args_i["_t0"],
                                            args_i["_user_args"])
                        else:
                            raise TypeError("If using string-list based" +
                                            "collapse operator" +
                                            "time-dependence, " +
                                            "args must be a dictionary")
                    c_ops_shifted.append([c_ops[i][0], fn])
                else:
                    c_ops_shifted.append(c_ops[i])

    return H_shifted, c_ops_shifted, _args

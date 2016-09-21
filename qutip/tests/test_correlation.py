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

from qutip import _version2int
from numpy import trapz, linspace, pi
from numpy.testing import run_module_suite, assert_
import unittest
import warnings

from qutip import (correlation, destroy, coherent_dm, correlation_2op_2t,
                   fock, correlation_2op_1t, tensor, qeye, spectrum_ss,
                   spectrum_pi, correlation_ss, spectrum_correlation_fft,
                   spectrum, correlation_3op_2t, mesolve, Options,
                   Cubic_Spline)

# find Cython if it exists
try:
    import Cython
except:
    Cython_OK = False
else:
    Cython_OK = _version2int(Cython.__version__) >= _version2int('0.14')


def test_compare_solvers_coherent_state_legacy():
    """
    correlation: legacy me and es for oscillator in coherent initial state
    """

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]
    rho0 = coherent_dm(N, np.sqrt(4.0))
    taulist = np.linspace(0, 5.0, 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr1 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a,
                            solver="me")
        corr2 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a,
                            solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_compare_solvers_coherent_state_mees():
    """
    correlation: comparing me and es for oscillator in coherent initial state
    """

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]
    rho0 = coherent_dm(N, np.sqrt(4.0))

    taulist = np.linspace(0, 5.0, 100)
    corr1 = correlation_2op_2t(H, rho0, None, taulist, c_ops, a.dag(), a,
                               solver="me")
    corr2 = correlation_2op_2t(H, rho0, None, taulist, c_ops, a.dag(), a,
                               solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_compare_solvers_coherent_state_memc():
    """
    correlation: comparing me and mc for driven oscillator in fock state
    """

    N = 2
    a = destroy(N)
    H = a.dag() * a + a + a.dag()
    G1 = 0.75
    n_th = 2.00
    c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]
    psi0 = fock(N, 1)

    taulist = np.linspace(0, 1.0, 3)
    corr1 = correlation_2op_2t(H, psi0, [0, 0.5], taulist, c_ops, a.dag(), a,
                               solver="me")
    corr2 = correlation_2op_2t(H, psi0, [0, 0.5], taulist, c_ops, a.dag(), a,
                               solver="mc")

    # pretty lax criterion, but would otherwise require a quite long simulation
    # time
    assert_(abs(corr1 - corr2).max() < 0.2)


def test_compare_solvers_steadystate_legacy():
    """
    correlation: legacy me and es for oscillator in steady-state
    """

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]

    taulist = np.linspace(0, 5.0, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr1 = correlation(H, None, None, taulist, c_ops, a.dag(), a,
                            solver="me")
        corr2 = correlation(H, None, None, taulist, c_ops, a.dag(), a,
                            solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_compare_solvers_steadystate():
    """
    correlation: comparing me and es for oscillator in steady-state
    """

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]

    taulist = np.linspace(0, 5.0, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr1 = correlation_2op_1t(H, None, taulist, c_ops, a.dag(), a,
                                   solver="me")
        corr2 = correlation_2op_1t(H, None, taulist, c_ops, a.dag(), a,
                                   solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_spectrum_espi_legacy():
    """
    correlation: legacy spectrum from es and pi methods
    """

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * np.pi
    g = 0.1 * 2 * np.pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [np.sqrt(kappa * (1 + n_th)) * a,
             np.sqrt(kappa * n_th) * a.dag(),
             np.sqrt(gamma) * sm]

    wlist = 2 * np.pi * np.linspace(0.5, 1.5, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec1 = spectrum_ss(H, wlist, c_ops, a.dag(), a)
        spec2 = spectrum_pi(H, wlist, c_ops, a.dag(), a)

    assert_(max(abs(spec1 - spec2)) < 1e-3)


def test_spectrum_esfft():
    """
    correlation: comparing spectrum from es and fft methods
    """

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * np.pi
    g = 0.1 * 2 * np.pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [np.sqrt(kappa * (1 + n_th)) * a,
             np.sqrt(kappa * n_th) * a.dag(),
             np.sqrt(gamma) * sm]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tlist = np.linspace(0, 100, 2500)
        corr = correlation_ss(H, tlist, c_ops, a.dag(), a)
        wlist1, spec1 = spectrum_correlation_fft(tlist, corr)
        spec2 = spectrum_ss(H, wlist1, c_ops, a.dag(), a)

    assert_(max(abs(spec1 - spec2)) < 1e-3)


def test_spectrum_espi():
    """
    correlation: comparing spectrum from es and pi methods
    """

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * np.pi
    g = 0.1 * 2 * np.pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [np.sqrt(kappa * (1 + n_th)) * a,
             np.sqrt(kappa * n_th) * a.dag(),
             np.sqrt(gamma) * sm]

    wlist = 2 * pi * np.linspace(0.5, 1.5, 100)
    spec1 = spectrum(H, wlist, c_ops, a.dag(), a, solver='es')
    spec2 = spectrum(H, wlist, c_ops, a.dag(), a, solver='pi')

    assert_(max(abs(spec1 - spec2)) < 1e-3)


@unittest.skipIf(not Cython_OK, 'Cython not found or version too low.')
def test_H_str_list_td_corr():
    """
    correlation: comparing TLS emission corr., H td (str-list td format)
    """

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.57
    sm = destroy(2)
    args = {"t_off": 1, "tp": 0.5}
    H = [[2 * (sm+sm.dag()), "exp(-(t-t_off)**2 / (2*tp**2))"]]
    tlist = np.linspace(0, 5, 50)
    corr = correlation_3op_2t(H, fock(2, 0), tlist, tlist, [sm],
                              sm.dag(), sm.dag() * sm, sm, args=args)
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1]-tlist[0]) / (np.shape(tlist)[0]-1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) + \
        sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = np.trapz(
        mesolve(
            H, fock(2, 0), tlist, [sm], [sm.dag()*sm], args=args
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5*dt**2*(s1 + 2*s2 + 4*s3)) / exp_n_in**2
    )

    assert_(abs(g20-0.59) < 1e-2)


@unittest.skipIf(not Cython_OK, 'Cython not found or version too low.')
def test_H_np_list_td_corr():
    """
    correlation: comparing TLS emission corr., H td (np-list td format)
    """

    #from qutip.rhs_generate import rhs_clear

    #rhs_clear()

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.57
    sm = destroy(2)
    tp = 0.5
    t_off = 1
    tlist = np.linspace(0, 5, 50)
    H = [[2 * (sm + sm.dag()), np.exp(-(tlist - t_off) ** 2 / (2 * tp ** 2))]]
    corr = correlation_3op_2t(H, fock(2, 0), tlist, tlist, [sm],
                              sm.dag(), sm.dag() * sm, sm)
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1] - tlist[0]) / (np.shape(tlist)[0] - 1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) + \
         sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = np.trapz(
        mesolve(
            H, fock(2, 0), tlist, [sm], [sm.dag() * sm]
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5 * dt ** 2 * (s1 + 2 * s2 + 4 * s3)) / exp_n_in ** 2
    )

    assert_(abs(g20 - 0.59) < 1e-2)


def test_H_fn_list_td_corr():
    """
    correlation: comparing TLS emission corr., H td (fn-list td format)
    """

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.57
    sm = destroy(2)
    args = {"t_off": 1, "tp": 0.5}
    H = [[2 * (sm+sm.dag()),
          lambda t, args: np.exp(-(t-args["t_off"])**2 / (2*args["tp"]**2))]]
    tlist = linspace(0, 5, 50)
    corr = correlation_3op_2t(H, fock(2, 0), tlist, tlist, [sm],
                              sm.dag(), sm.dag() * sm, sm, args=args)
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1]-tlist[0]) / (np.shape(tlist)[0]-1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) + \
        sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = np.trapz(
        mesolve(
            H, fock(2, 0), tlist, [sm], [sm.dag()*sm], args=args
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5*dt**2*(s1 + 2*s2 + 4*s3)) / exp_n_in**2
    )

    assert_(abs(g20-0.59) < 1e-2)


def test_H_fn_td_corr():
    """
    correlation: comparing TLS emission corr., H td (fn td format)
    """

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.57
    sm = destroy(2)

    def H_func(t, args):
        return 2 * args["H0"] * np.exp(-2 * (t-1)**2)

    tlist = linspace(0, 5, 50)
    corr = correlation_3op_2t(H_func, fock(2, 0), tlist, tlist,
                              [sm], sm.dag(), sm.dag() * sm, sm,
                              args={"H0": sm+sm.dag()})
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1]-tlist[0]) / (np.shape(tlist)[0]-1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) +\
        sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = trapz(
        mesolve(
            H_func, fock(2, 0), tlist, [sm], [sm.dag()*sm],
            args={"H0": sm+sm.dag()}
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5*dt**2*(s1 + 2*s2 + 4*s3)) / exp_n_in**2
    )

    assert_(abs(g20-0.59) < 1e-2)


@unittest.skipIf(not Cython_OK, 'Cython not found or version too low.')
def test_c_ops_str_list_td_corr():
    """
    correlation: comparing 3LS emission corr., c_ops td (str-list td format)
    """

    # calculate zero-delay HOM cross-correlation, for incoherently pumped
    # 3LS ladder system g2ab[0]
    # with following parameters:
    #   gamma = 1, 99% initialized, tp = 0.5
    # Then: g2ab(0)~0.185
    tlist = np.linspace(0, 6, 20)
    ket0 = fock(3, 0)
    ket1 = fock(3, 1)
    ket2 = fock(3, 2)
    sm01 = ket0 * ket1.dag()
    sm12 = ket1 * ket2.dag()
    psi0 = fock(3, 2)

    tp = 1
    # define "pi" pulse as when 99% of population has been transferred
    Om = np.sqrt(-np.log(1e-2) / (tp * np.sqrt(np.pi)))
    c_ops = [sm01,
             [sm12 * Om, "exp(-(t - t_off) ** 2 / (2 * tp ** 2))"]]
    args = {"tp": tp, "t_off": 2}
    H = qeye(3) * 0
    # HOM cross-correlation depends on coherences (g2[0]=0)
    c1 = correlation_2op_2t(H, psi0, tlist, tlist, c_ops,
                            sm01.dag(), sm01, args=args)
    c2 = correlation_2op_2t(H, psi0, tlist, tlist, c_ops,
                            sm01.dag(), sm01, args=args, reverse=True)
    n = mesolve(
        H, psi0, tlist, c_ops, [sm01.dag() * sm01], args=args
    ).expect[0]
    n_f = Cubic_Spline(tlist[0], tlist[-1], n)
    corr_ab = - c1 * c2 + np.array(
        [[n_f(t) * n_f(t + tau) for tau in tlist]
         for t in tlist])
    dt = tlist[1] - tlist[0]
    gab = abs(np.trapz(np.trapz(corr_ab, axis=0))) * dt ** 2

    assert_(abs(gab - 0.185) < 1e-2)


@unittest.skipIf(not Cython_OK, 'Cython not found or version too low.')
def test_np_str_list_td_corr():
    """
    correlation: comparing 3LS emission corr., c_ops td (np-list td format)
    """

    # calculate zero-delay HOM cross-correlation, for incoherently pumped
    # 3LS ladder system g2ab[0]
    # with following parameters:
    #   gamma = 1, 99% initialized, tp = 0.5
    # Then: g2ab(0)~0.185
    tlist = np.linspace(0, 6, 20)
    ket0 = fock(3, 0)
    ket1 = fock(3, 1)
    ket2 = fock(3, 2)
    sm01 = ket0 * ket1.dag()
    sm12 = ket1 * ket2.dag()
    psi0 = fock(3, 2)

    tp = 1
    t_off = 2
    # define "pi" pulse as when 99% of population has been transferred
    Om = np.sqrt(-np.log(1e-2) / (tp * np.sqrt(np.pi)))
    c_ops = [sm01,
             [sm12 * Om, np.exp(-(tlist - t_off) ** 2 / (2 * tp ** 2))]]
    H = qeye(3) * 0
    # HOM cross-correlation depends on coherences (g2[0]=0)
    c1 = correlation_2op_2t(H, psi0, tlist, tlist, c_ops,
                            sm01.dag(), sm01)
    c2 = correlation_2op_2t(H, psi0, tlist, tlist, c_ops,
                            sm01.dag(), sm01, reverse=True)
    n = mesolve(
        H, psi0, tlist, c_ops, [sm01.dag() * sm01]
    ).expect[0]
    n_f = Cubic_Spline(tlist[0], tlist[-1], n)
    corr_ab = - c1 * c2 + np.array(
        [[n_f(t) * n_f(t + tau) for tau in tlist]
         for t in tlist])
    dt = tlist[1] - tlist[0]
    gab = abs(np.trapz(np.trapz(corr_ab, axis=0))) * dt ** 2

    assert_(abs(gab - 0.185) < 1e-2)


def test_c_ops_fn_list_td_corr():
    """
    correlation: comparing 3LS emission corr., c_ops td (fn-list td format)
    """

    # calculate zero-delay HOM cross-correlation, for incoherently pumped
    # 3LS ladder system g2ab[0]
    # with following parameters:
    #   gamma = 1, 99% initialized, tp = 0.5
    # Then: g2ab(0)~0.185
    tlist = np.linspace(0, 6, 20)
    ket0 = fock(3, 0)
    ket1 = fock(3, 1)
    ket2 = fock(3, 2)
    sm01 = ket0 * ket1.dag()
    sm12 = ket1 * ket2.dag()
    psi0 = fock(3, 2)

    tp = 1
    # define "pi" pulse as when 99% of population has been transferred
    Om = np.sqrt(-np.log(1e-2) / (tp * np.sqrt(np.pi)))
    c_ops = [sm01,
             [sm12 * Om,
              lambda t, args:
                    np.exp(-(t - args["t_off"]) ** 2 / (2 * args["tp"] ** 2))]]
    args = {"tp": tp, "t_off": 2}
    H = qeye(3) * 0
    # HOM cross-correlation depends on coherences (g2[0]=0)
    c1 = correlation_2op_2t(H, psi0, tlist, tlist, c_ops,
                            sm01.dag(), sm01, args=args)
    c2 = correlation_2op_2t(H, psi0, tlist, tlist, c_ops,
                            sm01.dag(), sm01, args=args, reverse=True)
    n = mesolve(
        H, psi0, tlist, c_ops, [sm01.dag() * sm01], args=args
    ).expect[0]
    n_f = Cubic_Spline(tlist[0], tlist[-1], n)
    corr_ab = - c1 * c2 + np.array(
        [[n_f(t) * n_f(t + tau) for tau in tlist]
         for t in tlist])
    dt = tlist[1] - tlist[0]
    gab = abs(np.trapz(np.trapz(corr_ab, axis=0))) * dt ** 2

    assert_(abs(gab - 0.185) < 1e-2)


@unittest.skipIf(not Cython_OK, 'Cython not found or version too low.')
def test_str_list_td_corr():
    """
    correlation: comparing TLS emission corr. (str-list td format)
    """

    # both H and c_ops are time-dependent

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.85
    sm = destroy(2)
    args = {"t_off": 1, "tp": 0.5}
    tlist = np.linspace(0, 5, 50)
    H = [[2 * (sm + sm.dag()), "exp(-(t-t_off)**2 / (2*tp**2))"]]
    c_ops = [sm, [sm.dag() * sm * 2, "exp(-(t-t_off)**2 / (2*tp**2))"]]
    corr = correlation_3op_2t(H, fock(2, 0), tlist, tlist, [sm],
                              sm.dag(), sm.dag() * sm, sm, args=args)
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1] - tlist[0]) / (np.shape(tlist)[0] - 1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) + \
         sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = np.trapz(
        mesolve(
            H, fock(2, 0), tlist, c_ops, [sm.dag() * sm], args=args
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5 * dt ** 2 * (s1 + 2 * s2 + 4 * s3)) / exp_n_in ** 2
    )

    assert_(abs(g20 - 0.85) < 1e-2)


@unittest.skipIf(not Cython_OK, 'Cython not found or version too low.')
def test_np_list_td_corr():
    """
    correlation: comparing TLS emission corr. (np-list td format)
    """

    # both H and c_ops are time-dependent

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.85
    sm = destroy(2)
    t_off = 1
    tp = 0.5
    tlist = np.linspace(0, 5, 50)
    H = [[2 * (sm + sm.dag()), np.exp(-(tlist-t_off)**2 / (2*tp**2))]]
    c_ops = [sm, [sm.dag() * sm * 2, np.exp(-(tlist-t_off)**2 / (2*tp**2))]]
    corr = correlation_3op_2t(H, fock(2, 0), tlist, tlist, [sm],
                              sm.dag(), sm.dag() * sm, sm)
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1] - tlist[0]) / (np.shape(tlist)[0] - 1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) + \
         sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = np.trapz(
        mesolve(
            H, fock(2, 0), tlist, c_ops, [sm.dag() * sm]
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5 * dt ** 2 * (s1 + 2 * s2 + 4 * s3)) / exp_n_in ** 2
    )

    assert_(abs(g20 - 0.85) < 1e-2)


def test_fn_list_td_corr():
    """
    correlation: comparing TLS emission corr. (fn-list td format)
    """

    # both H and c_ops are time-dependent

    # calculate emission zero-delay second order correlation, g2[0], for TLS
    # with following parameters:
    #   gamma = 1, omega = 2, tp = 0.5
    # Then: g2(0)~0.85
    sm = destroy(2)
    args = {"t_off": 1, "tp": 0.5}
    H = [[2 * (sm + sm.dag()),
          lambda t, args:
                np.exp(-(t - args["t_off"]) ** 2 / (2 * args["tp"] ** 2))]]
    c_ops = [sm, [sm.dag() * sm * 2,
                  lambda t, args:
                    np.exp(-(t - args["t_off"]) ** 2 / (2 * args["tp"] ** 2))]]
    tlist = np.linspace(0, 5, 50)
    corr = correlation_3op_2t(H, fock(2, 0), tlist, tlist, [sm],
                              sm.dag(), sm.dag() * sm, sm, args=args)
    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1] - tlist[0]) / (np.shape(tlist)[0] - 1)
    s1 = corr[0, 0] + corr[-1, 0] + corr[0, -1] + corr[-1, -1]
    s2 = sum(corr[1:-1, 0]) + sum(corr[1:-1, -1]) + \
         sum(corr[0, 1:-1]) + sum(corr[-1, 1:-1])
    s3 = sum(corr[1:-1, 1:-1])

    exp_n_in = np.trapz(
        mesolve(
            H, fock(2, 0), tlist, c_ops, [sm.dag() * sm], args=args
        ).expect[0], tlist
    )
    # factor of 2 from negative time correlations
    g20 = abs(
        sum(0.5 * dt ** 2 * (s1 + 2 * s2 + 4 * s3)) / exp_n_in ** 2
    )

    assert_(abs(g20 - 0.85) < 1e-2)


if __name__ == "__main__":
    run_module_suite()

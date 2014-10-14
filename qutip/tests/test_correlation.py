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
                   spectrum, correlation_3op_2t, mesolve)

# find Cython if it exists
try:
    import Cython
except:
    Cython_found = 0
else:
    Cython_found = 1


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr1 = correlation_2op_2t(H, rho0, None, taulist, c_ops, a.dag(), a,
                                   solver="me")
        corr2 = correlation_2op_2t(H, rho0, None, taulist, c_ops, a.dag(), a,
                                   solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_compare_solvers_coherent_state_memc():
    """
    correlation: comparing me and mc for driven oscillator in ground state
    """

    N = 20
    a = destroy(N)
    H = a.dag() * a + a + a.dag()
    G1 = 0.75
    n_th = 2.00
    c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]
    psi0 = fock(N, 0)

    taulist = np.linspace(0, 1.0, 5)
    corr1 = correlation_2op_2t(H, psi0, [0], taulist, c_ops, a.dag(), a,
                               solver="me")[0]
    corr2 = correlation_2op_2t(H, psi0, [0], taulist, c_ops, a.dag(), a,
                               solver="mc")[0]

    assert_(max(abs(corr1 - corr2)) < 5e-2)


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


@unittest.skipIf(_version2int(Cython.__version__) < _version2int('0.14') or
                 Cython_found == 0, 'Cython not found or version too low.')
def test_str_list_td_corr():
    """
    correlation: comparing TLS emission correlations (str-list td format)
    """

    # calculate emission zero-delay second order correlation, g2(0), for TLS
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

    assert_(abs(g20-0.57) < 1e-1)


def test_fn_list_td_corr():
    """
    correlation: comparing TLS emission correlations (fn-list td format)
    """

    # calculate emission zero-delay second order correlation, g2(0), for TLS
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

    assert_(abs(g20-0.57) < 1e-1)


def test_fn_td_corr():
    """
    correlation: comparing TLS emission correlations (fn td format)
    """

    # calculate emission zero-delay second order correlation, g2(0), for TLS
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

    assert_(abs(g20-0.57) < 1e-1)


if __name__ == "__main__":
    run_module_suite()

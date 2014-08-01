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

from numpy import trapz, linspace
from qutip import _version2int
from numpy.testing import run_module_suite, assert_
import unittest

from qutip import *

# find Cython if it exists
try:
    import Cython
except:
    Cython_found = 0
else:
    Cython_found = 1


def test_compare_solvers_coherent_state():
    "correlation: comparing me and es for oscillator in coherent initial state"

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [sqrt(G1 * (1 + n_th)) * a, sqrt(G1 * n_th) * a.dag()]
    rho0 = coherent_dm(N, sqrt(4.0))

    taulist = np.linspace(0, 5.0, 100)
    corr1 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a, solver="me")
    corr2 = correlation(H, rho0, None, taulist, c_ops, a.dag(), a, solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_compare_solvers_steadystate():
    "correlation: comparing me and es for oscillator in steady state"

    N = 20
    a = destroy(N)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2.00
    c_ops = [sqrt(G1 * (1 + n_th)) * a, sqrt(G1 * n_th) * a.dag()]

    taulist = np.linspace(0, 5.0, 100)
    corr1 = correlation(H, None, None, taulist, c_ops, a.dag(), a, solver="me")
    corr2 = correlation(H, None, None, taulist, c_ops, a.dag(), a, solver="es")

    assert_(max(abs(corr1 - corr2)) < 1e-4)


def test_spectrum():
    "correlation: comparing spectrum from eseries and fft methods"

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * pi
    g = 0.1 * 2 * pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [sqrt(kappa * (1 + n_th)) * a,
             sqrt(kappa * n_th) * a.dag(),
             sqrt(gamma) * sm]

    tlist = np.linspace(0, 100, 2500)
    corr = correlation_ss(H, tlist, c_ops, a.dag(), a)
    wlist1, spec1 = spectrum_correlation_fft(tlist, corr)
    spec2 = spectrum_ss(H, wlist1, c_ops, a.dag(), a)

    assert_(max(abs(spec1 - spec2)) < 1e-3)


def test_spectrum():
    "correlation: comparing spectrum from eseries and pseudo-inverse methods"

    # use JC model
    N = 4
    wc = wa = 1.0 * 2 * pi
    g = 0.1 * 2 * pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
    c_ops = [sqrt(kappa * (1 + n_th)) * a,
             sqrt(kappa * n_th) * a.dag(),
             sqrt(gamma) * sm]

    wlist = 2 * pi * np.linspace(0.5, 1.5, 100)
    spec1 = spectrum_ss(H, wlist, c_ops, a.dag(), a)
    spec2 = spectrum_pi(H, wlist, c_ops, a.dag(), a)

    assert_(max(abs(spec1 - spec2)) < 1e-3)

@unittest.skipIf(_version2int(Cython.__version__) < _version2int('0.14') or
                 Cython_found == 0, 'Cython not found or version too low.')
def test_cython_td_corr():
    "correlation: comparing TLS emission correlations (cython td format)"

    # calculate emission zero-delay second order correlation, g2(0), directly from the photocount distribution
    # using mcsolve and make sure this equals the result expected from calculated integrating
    # the field flux correlations

    # generate TLS with following system parameters
    gamma, tp, E0, npts = 1, 0.5, 2, 50
    t_offset = 2*tp
    tmax = 2*t_offset + 3/gamma

    sm = destroy(2)

    args = {"t_off":t_offset,"tp":tp}
    H = [[E0*(sm + sm.dag()),"exp(-(t-t_off)**2/(2*tp**2))"]]

    tlist = linspace(0,tmax,npts)
    corr = correlation_4op_2t(H, fock(2,0), tlist, tlist, [sqrt(gamma)*sm], sm.dag(), sm.dag(), sm, sm, args=args)

    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1]-tlist[0])/(shape(tlist)[0]-1)
    s1 = corr[0,0] + corr[-1,0] + corr[0,-1] + corr[-1,-1]
    s2 = sum(corr[1:-1,0]) + sum(corr[1:-1,-1]) + sum(corr[0,1:-1]) + sum(corr[-1,1:-1])
    s3 = sum(corr[1:-1,1:-1])

    # calculate zero delay correlation from QRT
    exp_n_in = trapz(mesolve(H, fock(2,0), tlist, [sqrt(gamma)*sm], [sm.dag()*sm], args=args).expect[0],tlist)
    g20_corr = abs(sum(0.5*dt**2*(s1 + 2*s2 + 4*s3))/exp_n_in**2) # factor of 2 from negative time correlations

    tlist_mc = linspace(0,tmax,2)
    result_mc = mcsolve(H, fock(2,0), tlist_mc, [sqrt(gamma)*sm], [], ntraj=10000, args=args)

    # bin the collapse events to generate Pm
    ncollapse = [result_mc.col_times[i].size for i in range(result_mc.col_times.size)]
    Pm = numpy.histogram(ncollapse, bins=range(60), density=True)[0]

    # calculate zero delay correlation from mcsolve
    g20_mc = sum([p*m*(m-1) for m,p in enumerate(Pm)])/sum([p*m for m,p in enumerate(Pm)])**2

    assert_(abs(g20_mc - g20_corr) < 2e-1)

def test_fn_td_corr():
    "correlation: comparing TLS emission correlations (fn td format)"

    # calculate emission zero-delay second order correlation, g2(0), directly from the photocount distribution
    # using mcsolve and make sure this equals the result expected from calculated integrating
    # the field flux correlations

    # generate TLS with following system parameters
    gamma, tp, E0, npts = 1, 0.5, 2, 50
    t_offset = 2*tp
    tmax = 2*t_offset + 3/gamma

    sm = destroy(2)

    args = {"t_off":t_offset,"tp":tp}
    H = [[E0*(sm + sm.dag()),lambda t,args: exp(-(t-args["t_off"])**2/(2*args["tp"]**2))]]

    tlist = linspace(0,tmax,npts)
    corr = correlation_4op_2t(H, fock(2,0), tlist, tlist, [sqrt(gamma)*sm], sm.dag(), sm.dag(), sm, sm, args=args)

    # integrate w/ 2D trapezoidal rule
    dt = (tlist[-1]-tlist[0])/(shape(tlist)[0]-1)
    s1 = corr[0,0] + corr[-1,0] + corr[0,-1] + corr[-1,-1]
    s2 = sum(corr[1:-1,0]) + sum(corr[1:-1,-1]) + sum(corr[0,1:-1]) + sum(corr[-1,1:-1])
    s3 = sum(corr[1:-1,1:-1])

    # calculate zero delay correlation from QRT
    exp_n_in = trapz(mesolve(H, fock(2,0), tlist, [sqrt(gamma)*sm], [sm.dag()*sm], args=args).expect[0],tlist)
    g20_corr = abs(sum(0.5*dt**2*(s1 + 2*s2 + 4*s3))/exp_n_in**2) # factor of 2 from negative time correlations

    tlist_mc = linspace(0,tmax,2)
    H = [[E0*(sm + sm.dag()),"exp(-(t-t_off)**2/(2*tp**2))"]] # mcsolve doesn't seem to like function references when
                                                              # called in this test format
    result_mc = mcsolve(H, fock(2,0), tlist_mc, [sqrt(gamma)*sm], [], ntraj=10000, args=args)

    # bin the collapse events to generate Pm
    ncollapse = [result_mc.col_times[i].size for i in range(result_mc.col_times.size)]
    Pm = numpy.histogram(ncollapse, bins=range(60), density=True)[0]

    # calculate zero delay correlation from mcsolve
    g20_mc = sum([p*m*(m-1) for m,p in enumerate(Pm)])/sum([p*m for m,p in enumerate(Pm)])**2

    assert_(abs(g20_mc - g20_corr) < 2e-1)

if __name__ == "__main__":
    run_module_suite()

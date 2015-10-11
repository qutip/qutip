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
from numpy.testing import assert_,  run_module_suite

from qutip import ssesolve, destroy, coherent, mesolve, parallel_map


def test_ssesolve_photocurrent():
    "Stochastic: ssesolve: photo-current"
    tol = 0.01

    N = 4
    gamma = 0.25
    ntraj = 25
    nsubsteps = 100
    a = destroy(N)

    H = a.dag() * a
    psi0 = coherent(N, 0.5)
    sc_ops = [np.sqrt(gamma) * a]
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 2.5, 50)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops)
    res = ssesolve(H, psi0, times, sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='photocurrent', store_measurement=True,
                   map_func=parallel_map)

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops))
                 for m in res.measurement]))


def test_ssesolve_homodyne():
    "Stochastic: ssesolve: homodyne"
    tol = 0.01

    N = 4
    gamma = 0.25
    ntraj = 25
    nsubsteps = 100
    a = destroy(N)

    H = a.dag() * a
    psi0 = coherent(N, 0.5)
    sc_ops = [np.sqrt(gamma) * a]
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 2.5, 50)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops)
    res = ssesolve(H, psi0, times, sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='homodyne', store_measurement=True,
                   map_func=parallel_map)

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops))
                 for m in res.measurement]))


def test_ssesolve_heterodyne():
    "Stochastic: ssesolve: heterodyne"
    tol = 0.01

    N = 4
    gamma = 0.25
    ntraj = 25
    nsubsteps = 100
    a = destroy(N)

    H = a.dag() * a
    psi0 = coherent(N, 0.5)
    sc_ops = [np.sqrt(gamma) * a]
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 2.5, 50)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops)
    res = ssesolve(H, psi0, times, sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='heterodyne', store_measurement=True,
                   map_func=parallel_map)

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops), 2)
                 for m in res.measurement]))


if __name__ == "__main__":
    run_module_suite()

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

from qutip import (ssesolve, destroy, coherent, mesolve, fock, qeye,
                   parallel_map, photocurrent_sesolve)

def f(t, args):
    return args["a"] * t

def test_smesolve_homodyne_methods():
    "Stochastic: smesolve: homodyne methods with single jump operator"

    def arccoth(x):
        return 0.5*np.log((1.+x)/(x-1.))

    th = 0.1 # Interaction parameter
    alpha = np.cos(th)
    beta = np.sin(th)
    gamma = 1.

    N = 30                 # number of Fock states
    Id = qeye(N)
    a = destroy(N)
    s = 0.5*((alpha+beta)*a + (alpha-beta)*a.dag())
    x = (a + a.dag()) * 2**-0.5
    H = Id + gamma * a * a.dag()
    sc_op = [s]
    e_op = [x, x*x]
    rho0 = fock(N,0)      # initial vacuum state

    T = 6.                   # final time
    # number of time steps for which we save the expectation values
    N_store = 200
    Nsub = 10
    tlist = np.linspace(0, T, N_store)
    ddt = (tlist[1]-tlist[0])

    #### No analytic solution for ssesolve, taylor15 with 500 substep
    sol = ssesolve(H, rho0, tlist, sc_op, e_op,
                   nsubsteps=1000, method='homodyne', solver='taylor1.5')
    y_an = (sol.expect[1]-sol.expect[0]*sol.expect[0].conj())


    list_methods_tol = [['euler-maruyama', 3e-2],
                        ['pc-euler', 5e-3],
                        ['pc-euler-2', 5e-3],
                        ['platen', 5e-3],
                        ['milstein', 5e-3],
                        ['milstein-imp', 5e-3],
                        ['taylor1.5', 5e-4],
                        ['taylor1.5-imp', 5e-4],
                        ['explicit1.5', 5e-4],
                        ['taylor2.0', 5e-4]]
    for n_method in list_methods_tol:
        sol = ssesolve(H, rho0, tlist, sc_op, e_op,
                       nsubsteps=Nsub, method='homodyne', solver = n_method[0])
        sol2 = ssesolve(H, rho0, tlist, sc_op, e_op, store_measurement=0,
                       nsubsteps=Nsub, method='homodyne', solver = n_method[0],
                       noise = sol.noise)
        sol3 = ssesolve(H, rho0, tlist, sc_op, e_op,
                        nsubsteps=Nsub*5, method='homodyne',
                        solver = n_method[0], tol=1e-8)
        err = 1/T * np.sum(np.abs(y_an - \
                    (sol.expect[1]-sol.expect[0]*sol.expect[0].conj())))*ddt
        err3 = 1/T * np.sum(np.abs(y_an - \
                    (sol3.expect[1]-sol3.expect[0]*sol3.expect[0].conj())))*ddt
        print(n_method[0], ': deviation =', err, err3,', tol =', n_method[1])
        assert_(err < n_method[1])
        # 5* more substep should decrease the error
        assert_(err3 < err)
        # just to check that noise is not affected by smesolve
        assert_(np.all(sol.noise == sol2.noise))
        assert_(np.all(sol.expect[0] == sol2.expect[0]))

    sol = ssesolve(H, rho0, tlist[:2], sc_op, e_op, noise=10, ntraj=2,
                    nsubsteps=Nsub, method='homodyne', solver='euler',
                    store_measurement=1)
    sol2 = ssesolve(H, rho0, tlist[:2], sc_op, e_op, noise=10, ntraj=2,
                    nsubsteps=Nsub, method='homodyne', solver='euler',
                    store_measurement=0)
    sol3 = ssesolve(H, rho0, tlist[:2], sc_op, e_op, noise=11, ntraj=2,
                    nsubsteps=Nsub, method='homodyne', solver='euler')
    # sol and sol2 have the same seed, sol3 differ.
    assert_(np.all(sol.noise == sol2.noise))
    assert_(np.all(sol.noise != sol3.noise))
    assert_(not np.all(sol.measurement[0] == 0.+0j))
    assert_(np.all(sol2.measurement[0] == 0.+0j))
    sol = ssesolve(H, rho0, tlist[:2], sc_op, e_op, noise=np.array([1,2]),
                   ntraj=2, nsubsteps=Nsub, method='homodyne', solver='euler')
    sol2 = ssesolve(H, rho0, tlist[:2], sc_op, e_op, noise=np.array([2,1]),
                   ntraj=2, nsubsteps=Nsub, method='homodyne', solver='euler')
    # sol and sol2 have the seed of traj 1 and 2 reversed.
    assert_(np.all(sol.noise[0,:,:,:] == sol2.noise[1,:,:,:]))
    assert_(np.all(sol.noise[1,:,:,:] == sol2.noise[0,:,:,:]))


def test_ssesolve_photocurrent():
    "Stochastic: photocurrent_sesolve"
    tol = 0.01

    N = 4
    gamma = 0.25
    ntraj = 25
    nsubsteps = 100
    a = destroy(N)

    H = [[a.dag() * a,f]]
    psi0 = coherent(N, 0.5)
    sc_ops = [np.sqrt(gamma) * a, np.sqrt(gamma) * a*0.5]
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 2.5, 50)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops, args={"a":2})
    res = photocurrent_sesolve(H, psi0, times, sc_ops, e_ops, ntraj=ntraj,
                              nsubsteps=nsubsteps, store_measurement=True,
                              map_func=parallel_map, args={"a":2})

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops))
                 for m in res.measurement]))


def test_ssesolve_homodyne():
    "Stochastic: ssesolve: homodyne, time-dependent H"
    tol = 0.01

    N = 4
    gamma = 0.25
    ntraj = 25
    nsubsteps = 100
    a = destroy(N)

    H = [[a.dag() * a,f]]
    psi0 = coherent(N, 0.5)
    sc_ops = [np.sqrt(gamma) * a, np.sqrt(gamma) * a*0.5]
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 2.5, 50)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops, args={"a":2})
    res = ssesolve(H, psi0, times, sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='homodyne', store_measurement=True,
                   map_func=parallel_map, args={"a":2})

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops))
                 for m in res.measurement]))


def test_ssesolve_heterodyne():
    "Stochastic: ssesolve: heterodyne, time-dependent H"
    tol = 0.01

    N = 4
    gamma = 0.25
    ntraj = 25
    nsubsteps = 100
    a = destroy(N)

    H = [[a.dag() * a,f]]
    psi0 = coherent(N, 0.5)
    sc_ops = [np.sqrt(gamma) * a, np.sqrt(gamma) * a*0.5]
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 2.5, 50)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops, args={"a":2})
    res = ssesolve(H, psi0, times, sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='heterodyne', store_measurement=True,
                   map_func=parallel_map, args={"a":2})

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops), 2)
                 for m in res.measurement]))


if __name__ == "__main__":
    run_module_suite()

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
from numpy.testing import assert_, run_module_suite

from qutip import (smesolve, mesolve, destroy, coherent, parallel_map, qeye, fock_dm)


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
    H = Id
    c_op = [gamma**0.5 * a]
    sc_op = [s]
    e_op = [x, x*x]
    rho0 = fock_dm(N,0)      # initial vacuum state
    
    T = 6.                   # final time          
    N_store = 200            # number of time steps for which we save the expectation values
    Nsub = 5
    tlist = np.linspace(0, T, N_store)
    ddt = (tlist[1]-tlist[0])

    #### Analytic solution
    y0 = 0.5
    A = (gamma**2 + alpha**2 * (beta**2 + 4*gamma) - 2*alpha*beta*gamma)**0.5
    B = arccoth((-4*alpha**2*y0 + alpha*beta - gamma)/A)
    y_an = (alpha*beta - gamma + A / np.tanh(0.5*A*tlist - B))/(4*alpha**2)
    

    list_methods_tol = [['euler-maruyama', 2e-2],
                        ['fast-euler-maruyama', 2e-2],
                        ['pc-euler', 2e-3],
                        ['milstein', 1e-3],
                        ['fast-milstein', 1e-3],
                        ['milstein-imp', 1e-3],
                        ['taylor15', 1e-4],
                        ['taylor15-imp', 1e-4]
                        ]
    for n_method in list_methods_tol:
        sol = smesolve(H, rho0, tlist, c_op, sc_op, e_op,
                       nsubsteps=Nsub, method='homodyne', solver = n_method[0])
        sol2 = smesolve(H, rho0, tlist, c_op, sc_op, e_op,
                       nsubsteps=Nsub, method='homodyne', solver = n_method[0],
                       noise = sol.noise)
        err = 1/T * np.sum(np.abs(y_an - (sol.expect[1]-sol.expect[0]*sol.expect[0].conj())))*ddt
        print(n_method[0], ': deviation =', err, ', tol =', n_method[1])
        assert_(err < n_method[1])
        assert_(np.all(sol.noise == sol2.noise))# just to check that noise is not affected by smesolve
        assert_(np.all(sol.expect[0] == sol2.expect[0]))

def test_ssesolve_photocurrent():
    "Stochastic: smesolve: photo-current"
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
    res = smesolve(H, psi0, times, [], sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='photocurrent', store_measurement=True,
                   map_func=parallel_map)

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops))
                 for m in res.measurement]))


def test_ssesolve_homodyne():
    "Stochastic: smesolve: homodyne"
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
    res = smesolve(H, psi0, times, [], sc_ops, e_ops,
                   ntraj=ntraj, nsubsteps=nsubsteps,
                   method='homodyne', store_measurement=True,
                   map_func=parallel_map)

    assert_(all([np.mean(abs(res.expect[idx] - res_ref.expect[idx])) < tol
                 for idx in range(len(e_ops))]))

    assert_(len(res.measurement) == ntraj)
    assert_(all([m.shape == (len(times), len(sc_ops))
                 for m in res.measurement]))


def test_ssesolve_heterodyne():
    "Stochastic: smesolve: heterodyne"
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
    res = smesolve(H, psi0, times, [], sc_ops, e_ops,
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

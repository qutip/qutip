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
from numpy.testing import assert_, run_module_suite, assert_allclose
from qutip import *


def test_td_brmesolve_basic():
    """
    td_brmesolve: passes all brmesolve tests
    """
    
    # Test #1
    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    times = np.linspace(0, 10, 100)
    H = delta/2 * sigmax() + epsilon/2 * sigmaz()
    psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
    c_ops = [np.sqrt(gamma) * sigmam()]
    a_ops = [[sigmax(),'{0}*(w >= 0)'.format(gamma)]]
    e_ops = [sigmax(), sigmay(), sigmaz()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)

    
    # Test #2
    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    times = np.linspace(0, 10, 100)
    H = delta/2 * sigmax() + epsilon/2 * sigmaz()
    psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
    c_ops = [np.sqrt(gamma) * sigmam()]
    e_ops = [sigmax(), sigmay(), sigmaz()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve([[H,'1']], psi0, times, [], e_ops, c_ops=c_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)

    # Test #3
    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    times = np.linspace(0, 10, 100)
    H = delta/2 * sigmax() + epsilon/2 * sigmaz()
    psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()
    c_ops = [np.sqrt(gamma) * sigmam(), np.sqrt(gamma) * sigmaz()]
    c_ops_brme = [np.sqrt(gamma) * sigmaz()]
    a_ops = [[sigmax(),'{0}*(w >= 0)'.format(gamma)]]
    e_ops = [sigmax(), sigmay(), sigmaz()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops,c_ops=c_ops_brme)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)

    # Test #4
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15

    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    c_ops = [np.sqrt(kappa) * a]
    a_ops = [[a + a.dag(),'{0} * (w >= 0)'.format(kappa)]]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)

    # Test #5
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15
    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    n_th = 1.5
    w_th = w0/np.log(1 + 1/n_th)

    aop_str = "(({0} + 1) * {1})*(w>=0) + (({0}+1)*{1}*exp(w / {2}))*(w<0)" \
        .format(n_th,kappa,w_th)
    
    c_ops = [np.sqrt(kappa * (n_th + 1)) * a, np.sqrt(kappa * n_th) * a.dag()]
    a_ops = [[a + a.dag(),aop_str]]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)

    # Test #6
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.25
    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    n_th = 1.5
    w_th = w0/np.log(1 + 1/n_th)

    aop_str = "(({0} + 1) * {1})*(w>=0) + (({0}+1)*{1}*exp(w / {2}))*(w<0)" \
        .format(n_th,kappa,w_th)

    c_ops = [np.sqrt(kappa * (n_th + 1)) * a, np.sqrt(kappa * n_th) * a.dag()]
    a_ops = [[a + a.dag(),aop_str]]
    e_ops = []

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    n_me = expect(a.dag() * a, res_me.states)
    n_brme = expect(a.dag() * a, res_brme.states)

    diff = abs(n_me - n_brme).max()
    assert_(diff < 1e-2)
    
    # Test #7
    N = 10
    kappa = 0.05
    a = tensor(destroy(N), identity(2))
    sm = tensor(identity(N), destroy(2))
    psi0 = ket2dm(tensor(basis(N, 1), basis(2, 0)))
    a_ops = [[(a + a.dag()),'{kappa} * (w >= 0)'.format(kappa=kappa)]]
    e_ops = [a.dag() * a, sm.dag() * sm]

    w0 = 1.0 * 2 * np.pi
    g = 0.05 * 2 * np.pi
    times = np.linspace(0, 2 * 2 * np.pi / g, 1000)

    c_ops = [np.sqrt(kappa) * a]
    H = w0 * a.dag() * a + w0 * sm.dag() * sm + \
        g * (a + a.dag()) * (sm + sm.dag())

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 5e-2)  # accept 5% error


def test_td_brmesolve_aop():
    """
    td_brmesolve: time-dependent a_ops
    """
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    a_ops = [[a+a.dag(), '{kappa}*exp(-t)*(w>=0)'.format(kappa=kappa)]]
    tlist = np.linspace(0, 10, 100)
    medata = brmesolve(H, psi0, tlist, a_ops, e_ops=[a.dag() * a])
    expt = medata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_(avg_diff < 1e-6)


def test_td_brmesolve_aop_tuple1():
    """
    td_brmesolve: time-dependent a_ops tuple of strings
    """
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    a_ops = [[a+a.dag(), ('{kappa}*(w>=0)'.format(kappa=kappa),'exp(-t)')]]
    tlist = np.linspace(0, 10, 100)
    medata = brmesolve(H, psi0, tlist, a_ops, e_ops=[a.dag() * a])
    expt = medata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_(avg_diff < 1e-6)


def test_td_brmesolve_aop_tuple2():
    """
    td_brmesolve: time-dependent a_ops tuple interp
    """
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    tlist = np.linspace(0, 10, 100)
    S = Cubic_Spline(0,10,np.exp(-tlist))
    a_ops = [[a+a.dag(), ('{kappa}*(w>=0)'.format(kappa=kappa), S)]]
    medata = brmesolve(H, psi0, tlist, a_ops, e_ops=[a.dag() * a])
    expt = medata.expect[0]
    actual_answer = 9.0 * np.exp(-kappa * (1.0 - np.exp(-tlist)))
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_(avg_diff < 1e-5)


def test_td_brmesolve_aop_tuple3():
    """
    td_brmesolve: time-dependent a_ops & c_ops interp
    """
    N = 10  # number of basis states to consider
    a = destroy(N)
    H = a.dag() * a
    psi0 = basis(N, 9)  # initial state
    kappa = 0.2  # coupling to oscillator
    tlist = np.linspace(0, 10, 100)
    S = Cubic_Spline(0,10,np.exp(-tlist))
    S_c = Cubic_Spline(0,10,np.sqrt(kappa*np.exp(-tlist)))
    a_ops = [[a+a.dag(), ('{kappa}*(w>=0)'.format(kappa=kappa), S)]]
    medata = brmesolve(H, psi0, tlist, a_ops, e_ops=[a.dag() * a], c_ops=[[a,S_c]])
    expt = medata.expect[0]
    actual_answer = 9.0 * np.exp(-2*kappa * (1.0 - np.exp(-tlist)))
    avg_diff = np.mean(abs(actual_answer - expt) / actual_answer)
    assert_(avg_diff < 1e-5)

    
def test_td_brmesolve_nonherm_eops():
    """
    td_brmesolve: non-Hermitian e_ops check
    """
    N = 5
    a = destroy(N)
    rnd = np.random.random() +1j*np.random.random()
    H = a.dag()*a + rnd*a +np.conj(rnd)*a.dag()
    H2 = [[H,'1']]
    psi0 = basis(N,2)
    tlist = np.linspace(0,10,10)
    me = mesolve(H,psi0,tlist,c_ops=[],e_ops=[a],progress_bar=True)
    br = brmesolve(H2,psi0,tlist,a_ops=[],e_ops=[a],progress_bar=True)
    assert_(np.max(np.abs(me.expect[0]-br.expect[0])) < 1e-4)

    
def test_td_brmesolve_states():
    """
    td_brmesolve: states check
    """
    N = 5
    a = destroy(N)
    rnd = np.random.random() +1j*np.random.random()
    H = a.dag()*a + rnd*a +np.conj(rnd)*a.dag()
    H2 = [[H,'1']]
    psi0 = fock_dm(N,2)
    tlist = np.linspace(0,10,10)
    me = mesolve(H,psi0,tlist,c_ops=[],e_ops=[],progress_bar=True)
    br = brmesolve(H2,psi0,tlist,a_ops=[],e_ops=[],progress_bar=True)
    assert_(np.max([np.abs((me.states[kk]-br.states[kk]).full()).max() 
                for kk in range(len(tlist))]) < 1e-5)


def test_td_brmesolve_split_ops1():
    """
    td_brmesolve: split ops #1
    """
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15

    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    H2 = [[w0 * a.dag() * a + g * (a + a.dag()),'1']]
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())
    c_ops = [np.sqrt(kappa) * a, np.sqrt(kappa) * a]
    a_ops = [[ (a, a.dag()), ('{0} * (w >= 0)'.format(kappa), '1', '1') ] , 
                [a+a.dag(), '{0} * (w >= 0)'.format(kappa)]]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)
    
def test_td_brmesolve_split_ops2():
    """
    td_brmesolve: split ops #2
    """
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15

    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    H2 = [[w0 * a.dag() * a + g * (a + a.dag()),'1']]
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())
    c_ops = [np.sqrt(kappa) * a, np.sqrt(4*kappa) * a]
    a_ops = [[a+a.dag(), '{0} * (w >= 0)'.format(kappa)], [ (a, a.dag(), a, a.dag()), 
                    ('{0} * (w >= 0)'.format(kappa), '1', '1', '1', '1') ]]

    e_ops = [a.dag() * a, a + a.dag()]
    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def test_td_brmesolve_split_ops3():
    """
    td_brmesolve: split ops, Cubic_Spline td-terms
    """
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.15

    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    H2 = [[w0 * a.dag() * a + g * (a + a.dag()),'1']]
    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())
    
    S1 = Cubic_Spline(times[0],times[-1], np.ones_like(times))
    S2 = Cubic_Spline(times[0],times[-1], np.ones_like(times))
    
    c_ops = [np.sqrt(kappa) * a, np.sqrt(kappa) * a, np.sqrt(kappa) * a]
    a_ops = [ [a+a.dag(), '{0} * (w >= 0)'.format(kappa)],  [ (a, a.dag()), ('{0} * (w >= 0)'.format(kappa), S1, S2) ]]
    c_ops_br = [np.sqrt(kappa) * a]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops, c_ops_br)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)


def test_td_brmesolve_split_ops4():
    """
    td_brmesolve: split ops, multiple
    """
    N = 10
    w0 = 1.0 * 2 * np.pi
    g = 0.05 * w0
    kappa = 0.05

    times = np.linspace(0, 25, 1000)
    a = destroy(N)
    H = w0 * a.dag() * a + g * (a + a.dag())
    H2 = [[w0 * a.dag() * a + g * (a + a.dag()),'1']]

    psi0 = ket2dm((basis(N, 4) + basis(N, 2) + basis(N, 0)).unit())

    c_ops = [np.sqrt(kappa) * a, np.sqrt(kappa) * a, np.sqrt(kappa) * a]
    a1 = [(a, a.dag()), ('{0} * (w >= 0)'.format(kappa), '1', '1')]
    a2 = [a+a.dag(),'{0} * (w >= 0)'.format(kappa)]
    a3 = [(a, a.dag()), ('{0} * (w >= 0)'.format(kappa), '1', '1')]
    a_ops = [a1, a2, a3]
    e_ops = [a.dag() * a, a + a.dag()]

    res_me = mesolve(H, psi0, times, c_ops, e_ops)
    res_brme = brmesolve(H, psi0, times, a_ops, e_ops)

    for idx, e in enumerate(e_ops):
        diff = abs(res_me.expect[idx] - res_brme.expect[idx]).max()
        assert_(diff < 1e-2)

def test_td_brmesolve_args():
    """
    td_brmesolve: Hamiltonian args
    """
    N = 10
    a = tensor(destroy(N), identity(2))
    sm = tensor(identity(N), destroy(2))
    psi0 = ket2dm(tensor(basis(N, 1), basis(2, 0)))
    e_ops = [a.dag() * a, sm.dag() * sm]

    w0 = 1.0 * 2 * np.pi
    g = 0.75 * 2 * np.pi
    kappa = 0.05
    times = np.linspace(0, 5 * 2 * np.pi / g, 1000)

    a_ops = [[(a + a.dag()),'{k}*(w > 0)'.format(k=kappa)]]

    c_ops = [np.sqrt(kappa) * a]
    H = w0 * a.dag() * a + w0 * sm.dag() * sm + g * (a + a.dag()) * (sm + sm.dag())

    brme1 = brmesolve(H, psi0, times, a_ops, e_ops)

    H2= [[w0 * a.dag() * a + w0 * sm.dag() * sm + g * (a + a.dag()) * (sm + sm.dag()),'ii']]

    brme2 = brmesolve(H2, psi0, times, a_ops, e_ops, args={'ii': 1})

    assert_allclose(brme2.expect[0], brme1.expect[0])
    assert_allclose(brme2.expect[1], brme1.expect[1])

if __name__ == "__main__":
    run_module_suite()

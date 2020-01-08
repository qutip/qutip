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
from numpy.testing import assert_, assert_equal, run_module_suite
from qutip import *


def testPropHO():
    "Propagator: HO ('single mode')"
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H,1, unitary_mode='single')
    U2 = (-1j*H).expm()
    assert_(np.abs((U-U2).full()).max() < 1e-4)

def testPropHOB():
    "Propagator: HO ('batch mode')"
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H,1)
    U2 = (-1j*H).expm()
    assert_(np.abs((U-U2).full()).max() < 1e-4)

def testPropHOPar():
    "Propagator: HO parallel"
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H,1, parallel=True)
    U2 = (-1j*H).expm()
    assert_(np.abs((U-U2).full()).max() < 1e-4)


def testPropHOStrTd():
    "Propagator: str td format"
    a = destroy(5)
    H = a.dag()*a
    H = [H,[H,'cos(t)']]
    U = propagator(H,1, unitary_mode='single')
    U2 = propagator(H,1, parallel=True)
    U3 = propagator(H,1)
    assert_(np.abs((U-U2).full()).max() < 1e-4)
    assert_(np.abs((U-U3).full()).max() < 1e-4)


def func(t,*args):
    return np.cos(t)

def testPropHOFuncTd():
    "Propagator: func td format"
    a = destroy(5)
    H = a.dag()*a
    H = [H,[H,func]]
    U = propagator(H,1, unitary_mode='single')
    U2 = propagator(H,1, parallel=True)
    U3 = propagator(H,1)
    assert_(np.abs((U-U2).full()).max() < 1e-4)
    assert_(np.abs((U-U3).full()).max() < 1e-4)


def testPropHOSteady():
    "Propagator: steady state"
    a = destroy(5)
    H = a.dag()*a
    c_op_list = []
    kappa = 0.1
    n_th = 2
    rate = kappa * (1 + n_th)
    c_op_list.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    c_op_list.append(np.sqrt(rate) * a.dag())
    U = propagator(H,2*np.pi,c_op_list)
    rho_prop = propagator_steadystate(U)
    rho_ss = steadystate(H,c_op_list)
    assert_(np.abs((rho_prop-rho_ss).full()).max() < 1e-4)


def testPropHOSteadyPar():
    "Propagator: steady state parallel"
    a = destroy(5)
    H = a.dag()*a
    c_op_list = []
    kappa = 0.1
    n_th = 2
    rate = kappa * (1 + n_th)
    c_op_list.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    c_op_list.append(np.sqrt(rate) * a.dag())
    U = propagator(H,2*np.pi,c_op_list, parallel=True)
    rho_prop = propagator_steadystate(U)
    rho_ss = steadystate(H,c_op_list)
    assert_(np.abs((rho_prop-rho_ss).full()).max() < 1e-4)

def testPropHDims():
    "Propagator: preserve H dims (unitary_mode='single', parallel=False)"
    H = tensor([qeye(2),qeye(2)])
    U = propagator(H,1, unitary_mode='single')
    assert_equal(U.dims,H.dims)

if __name__ == "__main__":
    run_module_suite()

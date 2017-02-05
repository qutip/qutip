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
import scipy.interpolate as sint
from numpy.testing import assert_, assert_equal, run_module_suite
from qutip import *

    
def testInterpolate1():
    "Interpolate: Sine + noise (array)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+0.1*np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    assert_(np.max(np.abs(S2(x)-S1(x))) < 1e-9)

def testInterpolate2():
    "Interpolate: Sine + noise (point)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+0.1*np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    for k in range(x.shape[0]):
        assert_(np.abs(S2(x[k])-S1(x[k])) < 1e-9)

def testInterpolate3():
    "Interpolate: Complex sine + noise (array)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+0.1*np.random.randn(x.shape[0]) + \
        0.1j*np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    assert_(np.max(np.abs(S2(x)-S1(x))) < 1e-9)

def testInterpolate4():
    "Interpolate: Complex sine + noise (point)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+0.1*np.random.randn(x.shape[0]) + \
        0.1j*np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    for k in range(x.shape[0]):
        assert_(np.abs(S2(x[k])-S1(x[k])) < 1e-9)

def test_interpolate_evolve1():
    """
    Interpolate: sesolve str-based (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)']]
    H2 = [a.dag()*a,[a**2+a.dag()**2, S]]
    out1 = sesolve(H, psi0, tlist, [a.dag()*a]).expect[0]
    out2 = sesolve(H2, psi0, tlist, [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve2():
    """
    Interpolate: mesolve str-based (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)']]
    H2 = [a.dag()*a,[a**2+a.dag()**2, S]]
    out1 = mesolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mesolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve3():
    """
    Interpolate: mcsolve str-based (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)']]
    H2 = [a.dag()*a,[a**2+a.dag()**2, S]]
    out1 = mcsolve(H, psi0, tlist, [], [a.dag()*a],ntraj=500).expect[0]
    out2 = mcsolve(H2, psi0, tlist, [], [a.dag()*a],ntraj=500).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve4():
    """
    Interpolate: sesolve str + interp (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)'], [a**2+a.dag()**2,'0.25*cos(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,'0.25*cos(t)']]
    out1 = sesolve(H, psi0, tlist, [a.dag()*a]).expect[0]
    out2 = sesolve(H2, psi0, tlist, [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve5():
    """
    Interpolate: sesolve func + interp (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    func = lambda t, *args: 0.25*np.cos(t)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)'], [a**2+a.dag()**2,'0.25*cos(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,func]]
    out1 = sesolve(H, psi0, tlist, [a.dag()*a]).expect[0]
    out2 = sesolve(H2, psi0, tlist, [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve6():
    """
    Interpolate: mesolve str + interp (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock_dm(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)'], [a**2+a.dag()**2,'0.25*cos(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,'0.25*cos(t)']]
    out1 = mesolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mesolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve7():
    """
    Interpolate: mesolve func + interp (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    func = lambda t, *args: 0.25*np.cos(t)
    N = 10
    psi0 = fock_dm(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)'], [a**2+a.dag()**2,'0.25*cos(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,func]]
    out1 = mesolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mesolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve8():
    """
    Interpolate: mcsolve str + interp (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)'], [a**2+a.dag()**2,'0.25*cos(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,'0.25*cos(t)']]
    out1 = mcsolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mcsolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve9():
    """
    Interpolate: mcsolve func + interp (real)
    """
    tlist = np.linspace(0,5,50)
    y = 0.25*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    func = lambda t, *args: 0.25*np.cos(t)
    N = 10
    psi0 = fock(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.25*sin(t)'], [a**2+a.dag()**2,'0.25*cos(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,func]]
    out1 = mcsolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mcsolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve10():
    """
    Interpolate: mesolve str + interp (complex)
    """
    tlist = np.linspace(0,5,50)
    y = 0.1j*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    N = 10
    psi0 = fock_dm(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.1j*sin(t)'], [a**2+a.dag()**2,'-0.1j*sin(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,'-0.1j*sin(t)']]
    out1 = mesolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mesolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)

def test_interpolate_evolve11():
    """
    Interpolate: mesolve func + interp (complex)
    """
    tlist = np.linspace(0,5,50)
    y = 0.1j*np.sin(tlist)
    S = Cubic_Spline(tlist[0], tlist[-1], y)
    func = lambda t, *args: -0.1j*np.sin(t)
    N = 10
    psi0 = fock_dm(N,1)
    a = destroy(N)
    H = [a.dag()*a,[a**2+a.dag()**2,'0.1j*sin(t)'], [a**2+a.dag()**2,'-0.1j*sin(t)']]
    H2 = [a.dag()*a, [a**2+a.dag()**2, S], [a**2+a.dag()**2,func]]
    out1 = mesolve(H, psi0, tlist, [], [a.dag()*a]).expect[0]
    out2 = mesolve(H2, psi0, tlist, [], [a.dag()*a]).expect[0]
    print(np.max(np.abs(out1-out2)) < 1e-4)


if __name__ == "__main__":
    run_module_suite()

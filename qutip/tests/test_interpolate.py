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
    "Interpolation: Sine + noise (array)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    assert_(np.max(np.abs(S2(x)-S1(x))) < 1e-12)

def testInterpolate2():
    "Interpolation: Sine + noise (point)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    z = 2*np.pi*np.random.random()
    assert_(np.abs(S2(z)-S1(z)) < 1e-12)

def testInterpolate3():
    "Interpolation: Complex sine + noise (array)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+np.random.randn(x.shape[0]) + \
        1j*np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    assert_(np.max(np.abs(S2(x)-S1(x))) < 1e-12)

def testInterpolate4():
    "Interpolation: Complex sine + noise (point)"
    x = np.linspace(0,2*np.pi,200)
    y = np.sin(x)+np.random.randn(x.shape[0]) + \
        1j*np.random.randn(x.shape[0])
    S1 = Cubic_Spline(x[0],x[-1],y)
    S2 = sint.interp1d(x,y,'cubic')
    z = 2*np.pi*np.random.random()
    assert_(np.abs(S2(z)-S1(z)) < 1e-12)


if __name__ == "__main__":
    run_module_suite()

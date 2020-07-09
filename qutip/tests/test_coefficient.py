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

import pytest
import qutip as qt
import numpy as np
import qutip.coefficient as qtcoeff


def f(t, args):
    return np.exp(args["w"] * t * np.pi)


def g(t, args):
    return np.cos(args["w"] * t * np.pi)


args = {"w":1j}
tlist = np.linspace(0,1,101)
f_asarray = f(tlist, args)
g_asarray = g(tlist, args)
tlistlog = np.logspace(-2,0,501)
f_asarraylog = f(tlistlog, args)


@pytest.mark.parametrize(['base', 'kwargs', 'tol'], [
    pytest.param(f, {'args':args},
                 1e-10, id="func"),
    pytest.param(f_asarray, {'tlist':tlist},
                 1e-6,  id="array"),
    pytest.param(f_asarray, {'tlist':tlist, '_stepInterpolation':True},
                 1e-1, id="step_array"),
    pytest.param(f_asarraylog, {'tlist':tlistlog},
                 1e-6, id="nonlinear_array"),
    pytest.param(f_asarraylog, {'tlist':tlistlog, '_stepInterpolation':True},
                 1e-1, id="nonlinear_step_array"),
    pytest.param(qt.Cubic_Spline(0, 1, f_asarray), {},
                 1e-6, id="Cubic_Spline"),
    pytest.param("exp(w * t * pi)", {'args':args},
                 1e-10, id="string")
])
@pytest.mark.repeat(3)
def test_CoeffCreationCall(base, kwargs, tol):
    t = np.random.rand() * 0.9 + 0.05
    val = np.exp(1j * t * np.pi)
    coeff = qtcoeff.coefficient(base, **kwargs)
    assert np.allclose(coeff(t), val, rtol=tol)


@pytest.mark.parametrize(['base', 'kwargs', 'tol'], [
    pytest.param(f, {'args':args},
                 1e-10, id="func"),
    pytest.param("exp(w * t * pi)", {'args':args},
                 1e-10, id="string")
])
@pytest.mark.repeat(3)
def test_CoeffCallArgs(base, kwargs, tol):
    t = np.random.rand() * 0.9 + 0.05
    w = np.random.rand() + 0.5j
    val = np.exp(w * t * np.pi)
    coeff = qtcoeff.coefficient(f, **kwargs)
    assert np.allclose(coeff(t, {"w":w}), val, rtol=tol)


@pytest.mark.parametrize(['coeff'], [
    pytest.param(qtcoeff.coefficient(f, args=args), id="func"),
    pytest.param(qtcoeff.coefficient(f_asarray, tlist=tlist), id="array"),
    pytest.param(qtcoeff.coefficient(f_asarraylog, tlist=tlistlog),
                 id="logarray"),
    pytest.param(qtcoeff.coefficient(qt.Cubic_Spline(0, 1, f_asarray)),
                 id="Cubic_Spline"),
    pytest.param(qtcoeff.coefficient("exp(w * t * pi)", args=args),
                 id="string")
])
@pytest.mark.repeat(3)
def test_CoeffUnitaryTransform(coeff):
    t = np.random.rand() * 0.7 + 0.05
    dt = np.random.rand() * 0.2
    val = np.exp(1j * t * np.pi)
    val_dt = np.exp(1j * (t+dt) * np.pi)
    assert np.allclose(qtcoeff.norm(coeff)(t), np.abs(val)**2)
    assert np.allclose(qtcoeff.conj(coeff)(t), np.conj(val))
    assert np.allclose(qtcoeff.shift(coeff, dt)(t), val_dt)


@pytest.mark.parametrize(['coeff_left'], [
    pytest.param(qtcoeff.coefficient(f, args=args), id="func"),
    pytest.param(qtcoeff.coefficient(f_asarray, tlist=tlist), id="array"),
    pytest.param(qtcoeff.coefficient(qt.Cubic_Spline(0,1,f_asarray)),
                 id="Cubic_Spline"),
    pytest.param(qtcoeff.coefficient("exp(w * t * pi)", args=args),
                 id="string")
])
@pytest.mark.parametrize(['coeff_right'], [
    pytest.param(qtcoeff.coefficient(g, args=args), id="func"),
    pytest.param(qtcoeff.coefficient(g_asarray, tlist=tlist), id="array"),
    pytest.param(qtcoeff.coefficient(qt.Cubic_Spline(0,1,g_asarray)),
                 id="Cubic_Spline"),
    pytest.param(qtcoeff.coefficient("cos(w * t * pi)", args=args),
                 id="string")
])
@pytest.mark.repeat(3)
def test_CoeffOperation(coeff_left, coeff_right):
    t = np.random.rand() * 0.9 + 0.05
    val_l = np.exp(1j * t * np.pi)
    val_r = np.cos(1j * t * np.pi)
    coeff_sum = coeff_left + coeff_right
    assert np.allclose(coeff_sum(t), val_l + val_r)
    coeff_prod = coeff_left * coeff_right
    assert np.allclose(coeff_prod(t), val_l * val_r)


@pytest.mark.requires_cython
def test_CoeffReuse():
    coeff1 = qtcoeff.coefficient("cos(w * t * pi)", args={'w':3.})
    coeff2 = qtcoeff.coefficient("cos(w2*t * pi)", args={'w2':1.2})
    coeff3 = qtcoeff.coefficient("cos(  my_var * t*pi)", args={'my_var':-1.2})
    assert isinstance(coeff2, coeff1.__class__)
    assert isinstance(coeff3, coeff1.__class__)


@pytest.mark.slow
@pytest.mark.requires_cython
@pytest.mark.parametrize(['codestring', 'args', 'reference'], [
    pytest.param("cos(2*t)*cos(t*w1) + sin(w1*w2/2*t)*sin(t*w2)"
                 "- abs(exp(w1*w2*pi*0.25j)) ", {"w1":2,"w2":2},
                 lambda t: 0, id="long"),
    pytest.param("t*0.5 * (2) + 5j * -0.2j", {},
                 lambda t: t + 1, id="lotsofctes"),
    pytest.param("cos(t*vec[1])", {'vec':np.ones(2)},
                 lambda t: np.cos(t), id="realarray"),
    pytest.param("cos(t*vec[0])", {'vec':np.zeros(2)*1j},
                 lambda t: 1, id="complexarray"),
    pytest.param("cos(t*dictionary['key'])", {'dictionary':{'key':1}},
                 lambda t: np.cos(t), id="dictargs"),
    pytest.param("t + (0 if not 'something' else 1)", {},
                 lambda t: t + 1, id="branch")
])
def test_CoeffParsingStressTest(codestring, args, reference):
    t = np.random.rand() * 0.9 + 0.05
    coeff = qtcoeff.coefficient(codestring, args=args)
    assert np.allclose(coeff(t), reference(t))


@pytest.mark.requires_cython
@pytest.mark.repeat(3)
def test_CoeffReduce():
    t = np.random.rand() * 0.9 + 0.05
    coeff = qtcoeff.coefficient("exp(w * t * pi)", args={'w':1.0j})
    apad = coeff + qtcoeff.conj(coeff)
    reduced = qtcoeff.reduce(apad, {'w':1.0j})
    assert np.allclose(apad(t), reduced(t))

def test_pickling(...)

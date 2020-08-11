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
import pickle
import qutip as qt
import numpy as np
from qutip.core.coefficient import (coefficient, norm, conj, shift,
                                    reduce, CompilationOptions)


def f(t, args):
    return np.exp(args["w"] * t * np.pi)


def g(t, args):
    return np.cos(args["w"] * t * np.pi)


args = {"w": 1j}
tlist = np.linspace(0, 1, 101)
f_asarray = f(tlist, args)
g_asarray = g(tlist, args)
tlistlog = np.logspace(-2, 0, 501)
f_asarraylog = f(tlistlog, args)


def coeff_generator(style, func):
    """Make a Coefficient"""
    if func == "f":
        base = f
    else:
        base = g

    if style == "func":
        return coefficient(base, args=args)
    if style == "array":
        return coefficient(base(tlist, args), tlist=tlist)
    if style == "arraylog":
        return coefficient(base(tlistlog, args), tlist=tlistlog)
    if style == "spline":
        return coefficient(qt.Cubic_Spline(0, 1, base(tlist, args)))
    if style == "string" and func == "f":
        return coefficient("exp(w * t * pi)", args=args)
    if style == "string" and func == "g":
        return coefficient("cos(w * t * pi)", args=args)
    if style == "steparray":
        return coefficient(base(tlist, args), tlist=tlist,
                           _stepInterpolation=True)
    if style == "steparraylog":
        return coefficient(base(tlistlog, args), tlist=tlistlog,
                           _stepInterpolation=True)


@pytest.mark.parametrize(['base', 'kwargs', 'tol'], [
    pytest.param(f, {'args': args},
                 1e-10, id="func"),
    pytest.param(f_asarray, {'tlist': tlist},
                 1e-6,  id="array"),
    pytest.param(f_asarray, {'tlist': tlist, '_stepInterpolation': True},
                 1e-1, id="step_array"),
    pytest.param(f_asarraylog, {'tlist': tlistlog},
                 1e-6, id="nonlinear_array"),
    pytest.param(f_asarraylog, {'tlist': tlistlog, '_stepInterpolation': True},
                 1e-1, id="nonlinear_step_array"),
    pytest.param(qt.Cubic_Spline(0, 1, f_asarray), {},
                 1e-6, id="Cubic_Spline"),
    pytest.param("exp(w * t * pi)", {'args': args},
                 1e-10, id="string")
])
def test_CoeffCreationCall(base, kwargs, tol):
    CompilationOptions.recompile = True
    t = np.random.rand() * 0.9 + 0.05
    val = np.exp(1j * t * np.pi)
    coeff = coefficient(base, **kwargs)
    assert np.allclose(coeff(t), val, rtol=tol)
    CompilationOptions.recompile = False


@pytest.mark.parametrize(['base', 'kwargs', 'tol'], [
    pytest.param(f, {'args': args},
                 1e-10, id="func"),
    pytest.param("exp(w * t * pi)", {'args': args},
                 1e-10, id="string")
])
def test_CoeffCallArgs(base, kwargs, tol):
    t = np.random.rand() * 0.9 + 0.05
    w = np.random.rand() + 0.5j
    val = np.exp(w * t * np.pi)
    coeff = coefficient(f, **kwargs)
    assert np.allclose(coeff(t, {"w": w}), val, rtol=tol)


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("spline", id="Cubic_Spline"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
def test_CoeffUnitaryTransform(style):
    coeff = coeff_generator(style, "f")
    t = np.random.rand() * 0.7 + 0.05
    dt = np.random.rand() * 0.2
    val = coeff(t)
    val_dt = coeff(t+dt)
    assert np.allclose(norm(coeff)(t), np.abs(val)**2)
    assert np.allclose(conj(coeff)(t), np.conj(val))
    assert np.allclose(shift(coeff, dt)(t), val_dt)


@pytest.mark.parametrize(['style_left'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("spline", id="Cubic_Spline"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
@pytest.mark.parametrize(['style_right'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("spline", id="Cubic_Spline"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
def test_CoeffOperation(style_left, style_right):
    coeff_left = coeff_generator(style_left, "f")
    coeff_right = coeff_generator(style_right, "g")
    t = np.random.rand() * 0.9 + 0.05
    val_l = coeff_left(t)
    val_r = coeff_right(t)
    coeff_sum = coeff_left + coeff_right
    assert np.allclose(coeff_sum(t), val_l + val_r)
    coeff_prod = coeff_left * coeff_right
    assert np.allclose(coeff_prod(t), val_l * val_r)


@pytest.mark.requires_cython
def test_CoeffReuse():
    coeff1 = coefficient("cos(w * t * pi)", args={'w': 3.})
    coeff2 = coefficient("cos(w2*t * pi)", args={'w2': 1.2})
    coeff3 = coefficient("cos(  my_var * t*pi)", args={'my_var': -1.2})
    assert isinstance(coeff2, coeff1.__class__)
    assert isinstance(coeff3, coeff1.__class__)


@pytest.mark.requires_cython
def test_CoeffOptions():
    from itertools import combinations
    t = np.random.rand() * 0.9 + 0.05
    base = "1 + 1. + 1j"
    options = []
    options.append(CompilationOptions(accept_int=True))
    options.append(CompilationOptions(accept_float=False))
    options.append(CompilationOptions(no_types=True))
    options.append(CompilationOptions(use_cython=False))
    coeffs = [coefficient(base, compile_opt=opt) for opt in options]
    for coeff1, coeff2 in combinations(coeffs, 2):
        assert not isinstance(coeff1, coeff2.__class__)


@pytest.mark.requires_cython
@pytest.mark.parametrize(['codestring', 'args', 'reference'], [
    pytest.param("cos(2*t)*cos(t*w1) + sin(w1*w2/2*t)*sin(t*w2)"
                 "- abs(exp(w1*w2*pi*0.25j)) ", {"w1": 2, "w2": 2},
                 lambda t: 0, id="long"),
    pytest.param("t*0.5 * (2) + 5j * -0.2j", {},
                 lambda t: t + 1, id="lots_of_ctes"),
    pytest.param("cos(t*vec[1])", {'vec': np.ones(2)},
                 lambda t: np.cos(t), id="real_array_subscript"),
    pytest.param("cos(t*vec[0])", {'vec': np.zeros(2)*1j},
                 lambda t: 1, id="cplx_array_subscript"),
    pytest.param("cos(t*dictionary['key'])", {'dictionary': {'key': 1}},
                 lambda t: np.cos(t), id="dictargs"),
    pytest.param("cos(t*a); print(a)", {'a': 1},
                 lambda t: np.cos(t), id="print"),
    pytest.param("t + (0 if not 'something' else 1)", {},
                 lambda t: t + 1, id="branch")
])
def test_CoeffParsingStressTest(codestring, args, reference):
    CompilationOptions.recompile = True
    t = np.random.rand() * 0.9 + 0.05
    coeff = coefficient(codestring, args=args)
    assert np.allclose(coeff(t), reference(t))
    CompilationOptions.recompile = False


@pytest.mark.requires_cython
@pytest.mark.filterwarnings("error")
def test_manual_typing():
    CompilationOptions.recompile = True
    coeff = coefficient("my_list[0] + my_dict[5]",
                        args={"my_list": [1], "my_dict": {5: 2}},
                        args_ctypes={"my_list": "list", "my_dict": "dict"})
    assert coeff(0) == 3
    CompilationOptions.recompile = False


@pytest.mark.requires_cython
def test_advance_use():
    opt = CompilationOptions(recompile=True, extra_import="""
from qutip import basis
from qutip.core.data cimport CSR
from qutip.core.data.expect cimport expect_csr
""")
    csr = qt.num(3).data
    coeff = coefficient("expect_csr(op, op)",
                        args={"op": csr},
                        args_ctypes={"op": "CSR"},
                        compile_opt=opt)
    assert coeff(0) == 5.


@pytest.mark.requires_cython
def test_CoeffReduce():
    t = np.random.rand() * 0.9 + 0.05
    coeff = coefficient("exp(w * t * pi)", args={'w': 1.0j})
    apad = coeff + conj(coeff)
    reduced = reduce(apad, {'w': 1.0j})
    assert np.allclose(apad(t), reduced(t))


def _add(coeff):
    return coeff + coeff


def _pass(coeff):
    return coeff


def _mul(coeff):
    return coeff * coeff


def _shift(coeff):
    return shift(coeff, 0.05)


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("spline", id="Cubic_Spline"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
@pytest.mark.parametrize(['transform'], [
    pytest.param(_pass, id="single"),
    pytest.param(_add, id="sum"),
    pytest.param(_mul, id="prod"),
    pytest.param(norm, id="norm"),
    pytest.param(conj, id="conj"),
    pytest.param(_shift, id="shift"),
])
def test_Coeffpickle(style, transform):
    coeff = coeff_generator(style, "f")
    t = np.random.rand() * 0.85 + 0.05
    coeff = transform(coeff)
    coeff_pick = pickle.loads(pickle.dumps(coeff, -1))
    # Check for const case
    assert coeff(t) == coeff_pick(t)


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("spline", id="Cubic_Spline"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
@pytest.mark.parametrize(['transform'], [
    pytest.param(_pass, id="single"),
    pytest.param(_add, id="sum"),
    pytest.param(_mul, id="prod"),
    pytest.param(norm, id="norm"),
    pytest.param(conj, id="conj"),
    pytest.param(_shift, id="shift"),
])
def test_Coeffcopy(style, transform):
    coeff = coeff_generator(style, "f")
    t = np.random.rand() * 0.85 + 0.05
    coeff = transform(coeff)
    coeff_cp = coeff.copy()
    # Check for const case
    assert coeff(t) == coeff_cp(t)

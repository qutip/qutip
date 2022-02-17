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
import qutip
import numpy as np
import scipy.interpolate as interp
from functools import partial
from qutip.core.coefficient import (coefficient, norm, conj, shift,
                                    CompilationOptions,
                                    clean_compiled_coefficient
                                   )


# Ensure the latest version is tested
clean_compiled_coefficient(True)


def f(t, w):
    return np.exp(w * t * np.pi)


def g(t, w):
    return np.cos(w * t * np.pi)


def h(t, a, b):
    return a + b + t


def f_kw(t, w, **args):
    return f(t, w)


def g_kw(t, w, **args):
    return g(t, w)


def h_kw(t, a, b, **args):
    return h(t, a, b)


def f_qtv4(t, args):
    return f(t, args["w"])


def g_qtv4(t, args):
    return g(t, args["w"])


def h_qtv4(t, args):
    return h(t, args["a"], args["b"])


def _assert_eq_over_interval(coeff1, coeff2, rtol=1e-12, inside=False):
    "assert coeff1 == coeff2"
    "inside refer to the range covered by tlistlog: [0.01, 1]"
    ts = np.linspace(0.01, 1, 20)
    eps = 1e-12
    crit_times = [0.01+eps, 0.95, 1-eps]
    if not inside:
        crit_times += [-0.1, 0, eps, -eps, 1+eps, 1.1]
    c1 = [coeff1(t) for t in ts] + [coeff1(t) for t in crit_times]
    c2 = [coeff2(t) for t in ts] + [coeff2(t) for t in crit_times]
    np.testing.assert_allclose(c1, c2, rtol=rtol, atol=1e-15)


args = {"w": 1j}
tlist = np.linspace(0, 1, 101)
f_asarray = f(tlist, **args)
g_asarray = g(tlist, **args)
tlistlog = np.logspace(-2, 0, 501)
f_asarraylog = f(tlistlog, **args)


def coeff_generator(style, func):
    """Make a Coefficient"""
    if func == "f":
        base = f
    else:
        base = g

    if style == "func":
        return coefficient(base, args=args)
    if style == "array":
        return coefficient(base(tlist, **args), tlist=tlist)
    if style == "arraylog":
        return coefficient(base(tlistlog, **args), tlist=tlistlog)
    if style == "string" and func == "f":
        return coefficient("exp(w * t * pi)", args=args)
    if style == "string" and func == "g":
        return coefficient("cos(w * t * pi)", args=args)
    if style == "steparray":
        return coefficient(base(tlist, **args), tlist=tlist,
                           order=0)
    if style == "steparraylog":
        return coefficient(base(tlistlog, **args), tlist=tlistlog,
                           order=0)


@pytest.mark.parametrize(['base', 'kwargs', 'tol'], [
    pytest.param(f, {'args': args},
                 1e-10, id="func"),
    pytest.param(f_kw, {'args': args},
                 1e-10, id="func_keywords"),
    pytest.param(f_qtv4, {'args': args},
                 1e-10, id="func_qutip_v4"),
    pytest.param(f_asarray, {'tlist': tlist},
                 1e-6,  id="array"),
    pytest.param(f_asarray, {'tlist': tlist, 'order': 0},
                 1e-1, id="step_array"),
    pytest.param(f_asarraylog, {'tlist': tlistlog},
                 1e-6, id="nonlinear_array"),
    pytest.param(f_asarraylog, {'tlist': tlistlog, 'order': 0},
                 1e-1, id="nonlinear_step_array"),
    pytest.param("exp(w * t * pi)", {'args': args},
                 1e-10, id="string")
])
def test_CoeffCreationCall(base, kwargs, tol):
    opt = CompilationOptions(recompile=True)
    expected = lambda t: np.exp(1j * t * np.pi)
    coeff = coefficient(base, **kwargs, compile_opt=opt)
    _assert_eq_over_interval(coeff, expected, rtol=tol, inside=True)


@pytest.mark.parametrize(['base', 'kwargs', 'tol'], [
    pytest.param(f, {'args': args},
                 1e-10, id="func"),
    pytest.param(f_kw, {'args': args},
                 1e-10, id="func_keywords"),
    pytest.param(f_qtv4, {'args': args},
                 1e-10, id="func_qutip_v4"),
    pytest.param("exp(w * t * pi)", {'args': args},
                 1e-10, id="string")
])
def test_CoeffCallArgs(base, kwargs, tol):
    w = np.e + 0.5j
    expected = lambda t: np.exp(w * t * np.pi)
    coeff = coefficient(base, **kwargs)
    _assert_eq_over_interval(partial(coeff, w=w), expected, rtol=tol)


@pytest.mark.parametrize(['base', 'tol'], [
    pytest.param(h, 1e-10, id="func"),
    pytest.param(h_kw, 1e-10, id="func_keywords"),
    pytest.param(h_qtv4, 1e-10, id="func_qutip_v4"),
    pytest.param("a + b + t", 1e-10, id="string")
])
def test_CoeffCallArguments(base, tol):
    # Partial args update
    args = {"a": 1, "b": 1}
    a = np.e
    expected = lambda t: a + 1 + t
    coeff = coefficient(base, args=args)
    coeff = coeff.replace_arguments({"a": a})
    _assert_eq_over_interval(coeff, expected, rtol=tol)
    b = np.pi
    expected = lambda t: a + b + t
    coeff = coeff.replace_arguments(b=b)
    _assert_eq_over_interval(coeff, expected, rtol=tol)


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
@pytest.mark.parametrize(['transform', 'expected'], [
    pytest.param(norm, lambda val: np.abs(val)**2, id="norm"),
    pytest.param(conj, lambda val: np.conj(val), id="conj"),
])
def test_CoeffUnitaryTransform(style, transform, expected):
    coeff = coeff_generator(style, "f")
    _assert_eq_over_interval(transform(coeff), lambda t: expected(coeff(t)))


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
def test_CoeffShift(style):
    coeff = coeff_generator(style, "f")
    dt = np.e / 30
    _assert_eq_over_interval(shift(coeff, dt),
                             lambda t: coeff(t + dt))


@pytest.mark.parametrize(['style_left'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
@pytest.mark.parametrize(['style_right'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog")
])
@pytest.mark.parametrize(['oper'], [
    pytest.param(lambda a, b: a+b, id="sum"),
    pytest.param(lambda a, b: a*b, id="prod"),
])
def test_CoeffOperation(style_left, style_right, oper):
    coeff_left = coeff_generator(style_left, "f")
    coeff_right = coeff_generator(style_right, "g")
    _assert_eq_over_interval(
        oper(coeff_left, coeff_right),
        lambda t: oper(coeff_left(t), coeff_right(t))
    )


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
    base = "1 + 1. + 1j"
    with CompilationOptions(
        accept_int=None,
        accept_float=True,
        static_types=False,
        use_cython=True,
        try_parse=True,
        clean_on_error=False
    ):
        options = []
        options.append(CompilationOptions(accept_int=True))
        options.append(CompilationOptions(accept_float=False))
        options.append(CompilationOptions(static_types=True))
        options.append(CompilationOptions(try_parse=False))
    coeffs = [coefficient(base, compile_opt=opt) for opt in options]
    for coeff in coeffs:
        assert coeff(0) == 2+1j
    for coeffs1, coeffs2 in combinations(coeffs, 2):
        assert not isinstance(coeffs1, coeffs2.__class__)


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
    opt = CompilationOptions(recompile=True)
    coeff = coefficient(codestring, args=args, compile_opt=opt)
    _assert_eq_over_interval(coeff, reference)


@pytest.mark.requires_cython
@pytest.mark.filterwarnings("error")
def test_manual_typing():
    opt = CompilationOptions(recompile=True)
    coeff = coefficient("my_list[0] + my_dict[5]",
                        args={"my_list": [1], "my_dict": {5: 2}},
                        args_ctypes={"my_list": "list", "my_dict": "dict"},
                        compile_opt=opt)
    assert coeff(0) == 3


@pytest.mark.requires_cython
def test_advance_use():
    opt = CompilationOptions(
        recompile=True,
        extra_import="""
from qutip import basis
from qutip.core.data cimport CSR
from qutip.core.data.expect cimport expect_csr
""")
    csr = qutip.num(3).data
    coeff = coefficient("expect_csr(op, op)",
                        args={"op": csr},
                        args_ctypes={"op": "CSR"},
                        compile_opt=opt)
    assert coeff(0) == 5.


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
    coeff = transform(coeff)
    coeff_pick = pickle.loads(pickle.dumps(coeff, -1))
    _assert_eq_over_interval(coeff, coeff_pick)


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
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
    coeff = transform(coeff)
    coeff_cp = coeff.copy()
    _assert_eq_over_interval(coeff, coeff_cp)


@pytest.mark.parametrize('order', [0, 1, 2, 3])
def test_CoeffArray(order):
    tlist = np.linspace(0, 1, 101)
    y = np.exp((-1 + 1j) * tlist)
    coeff = coefficient(y, tlist=tlist, order=order)
    expected = coefficient(lambda t: np.exp((-1 + 1j) * t))
    _assert_eq_over_interval(coeff, expected, rtol=0.01**(order+0.8),
                             inside=True)
    dt = 1e-4
    t = 0.0225
    derr = (coeff(t+dt) - coeff(t-dt)) / (2*dt)
    derr2 = (coeff(t+dt) + coeff(t-dt) - 2 * coeff(t)) / (dt**2)
    derr3 = (coeff(t + 2*dt) - 2*coeff(t + dt)
             + 2*coeff(t - dt) -coeff(t - 2*dt)) / (12 * dt**3)
    derrs = [derr, derr2, derr3]
    for i in range(order):
        assert derrs[i] != 0
    for i in range(order, 3):
        assert derrs[i] == pytest.approx(0.0,  abs=0.0001)


def test_CoeffFromScipy():
    tlist = np.linspace(0, 1.01, 101)
    y = np.exp((-1 + 1j) * tlist)

    coeff = coefficient(y, tlist=tlist, order=3)
    from_scipy = coefficient(interp.CubicSpline(tlist, y))
    _assert_eq_over_interval(coeff, from_scipy, rtol=1e-8, inside=True)

    coeff = coefficient(y, tlist=tlist, order=3)
    from_scipy = coefficient(interp.make_interp_spline(tlist, y, k=3))
    _assert_eq_over_interval(coeff, from_scipy, rtol=1e-8, inside=True)



def test_func_signature():
    with qutip.core.CoreOptions(function_coefficient_style='pythonic'):
        assert coefficient(lambda t, args: args, args={'args': 1})(0) == 1
    with qutip.core.CoreOptions(function_coefficient_style='dict'):
        assert coefficient(lambda t, w: w['w'], args={'w': 1})(0) == 1

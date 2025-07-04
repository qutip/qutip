import pytest
import pickle
import qutip
import numpy as np
import scipy.interpolate as interp
from functools import partial
from qutip.core.coefficient import (coefficient, norm, conj, const,
                                    CompilationOptions, Coefficient,
                                    clean_compiled_coefficient,
                                    WARN_MISSING_MODULE,
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
    if style == "const":
        return const(2.0)


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


def test_coefficient_update_args():

    def f(t, a):
        return a

    coeff1 = coefficient(f, args={"a": 1})
    coeff2 = coefficient(coeff1, args={"a": 2})

    assert coeff2(0) == 2


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog"),
    pytest.param("const", id="constant"),
])
@pytest.mark.parametrize(['transform', 'expected'], [
    pytest.param(norm, lambda val: np.abs(val)**2, id="norm"),
    pytest.param(conj, lambda val: np.conj(val), id="conj"),
])
def test_CoeffUnitaryTransform(style, transform, expected):
    coeff = coeff_generator(style, "f")
    _assert_eq_over_interval(transform(coeff), lambda t: expected(coeff(t)))


def test_ConstantCoefficient():
    coeff = const(5.1)
    _assert_eq_over_interval(coeff, lambda t: 5.1)


@pytest.mark.parametrize(['style_left'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog"),
    pytest.param("const", id="constant"),
])
@pytest.mark.parametrize(['style_right'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog"),
    pytest.param("const", id="constant"),
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
    options = []
    options.append(CompilationOptions(accept_int=True))
    options.append(CompilationOptions(accept_float=True))
    options.append(CompilationOptions(static_types=True))
    options.append(CompilationOptions(try_parse=False))
    options.append(CompilationOptions(use_cython=False))
    coeffs = [coefficient(base, compile_opt=opt) for opt in options]
    for coeff in coeffs:
        assert coeff(0) == 2+1j
    for coeff1, coeff2 in combinations(coeffs, 2):
        assert not isinstance(coeff1, coeff2.__class__)


def test_warn_no_cython():
    option = CompilationOptions(use_cython=False)
    WARN_MISSING_MODULE[0] = 1
    with pytest.warns(UserWarning) as warning:
        coefficient("t", compile_opt=option)
    assert all(
        module in warning[0].message.args[0]
        for module in ["cython", "filelock", "setuptools"]
    )

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
@pytest.mark.filterwarnings(
    "ignore::qutip.core.coefficient.StringParsingWarning"
)
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
    opt = CompilationOptions(recompile=True, extra_import="""
from qutip import basis
from qutip.core.data cimport CSR
from qutip.core.data.expect cimport expect_csr
""")
    csr = qutip.num(3, dtype="CSR").data
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


@pytest.mark.parametrize(['style'], [
    pytest.param("func", id="func"),
    pytest.param("array", id="array"),
    pytest.param("arraylog", id="logarray"),
    pytest.param("string", id="string"),
    pytest.param("steparray", id="steparray"),
    pytest.param("steparraylog", id="steparraylog"),
    pytest.param("const", id="constant"),
])
@pytest.mark.parametrize(['transform'], [
    pytest.param(_pass, id="single"),
    pytest.param(_add, id="sum"),
    pytest.param(_mul, id="prod"),
    pytest.param(norm, id="norm"),
    pytest.param(conj, id="conj"),
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
    pytest.param("steparraylog", id="steparraylog"),
    pytest.param("const", id="constant"),
])
@pytest.mark.parametrize(['transform'], [
    pytest.param(_pass, id="single"),
    pytest.param(_add, id="sum"),
    pytest.param(_mul, id="prod"),
    pytest.param(norm, id="norm"),
    pytest.param(conj, id="conj"),
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


@pytest.mark.parametrize('imag', [True, False])
def test_CoeffFromScipyPPoly(imag):
    tlist = np.linspace(0, 1.01, 101)
    if imag:
        y = np.exp(-1j * tlist)
    else:
        y = np.exp(-1 * tlist)

    coeff = coefficient(y, tlist=tlist, order=3)
    from_scipy = coefficient(interp.CubicSpline(tlist, y))
    _assert_eq_over_interval(coeff, from_scipy, rtol=1e-8, inside=True)

    coeff = coefficient(y, tlist=tlist, order=3)
    from_scipy = coefficient(interp.make_interp_spline(tlist, y, k=3))
    _assert_eq_over_interval(coeff, from_scipy, rtol=1e-8, inside=True)

    coeff = coefficient(y, tlist=tlist, order=3, boundary_conditions="natural")
    from_scipy = coefficient(interp.make_interp_spline(tlist, y, k=3, bc_type="natural"))
    _assert_eq_over_interval(coeff, from_scipy, rtol=1e-8, inside=True)


@pytest.mark.parametrize('imag', [True, False])
def test_CoeffFromScipyBSpline(imag):
    tlist = np.linspace(-0.1, 1.1, 121)
    if imag:
        y = np.exp(-1j * tlist)
    else:
        y = np.exp(-1 * tlist)

    spline = interp.BSpline(tlist, y, 2)

    def func(t):
        return complex(spline(t))

    coverted = coefficient(spline)
    raw_scipy = coefficient(func)
    _assert_eq_over_interval(coverted, raw_scipy, rtol=1e-8, inside=True)


@pytest.mark.parametrize('map_func', [
    pytest.param(qutip.solver.parallel.parallel_map, id='parallel_map'),
    pytest.param(qutip.solver.parallel.loky_pmap, id='loky_pmap'),
])
@pytest.mark.requires_cython
def test_coefficient_parallel(map_func):
    otherwise_never_used = "np.log(np.exp(t + t + t))"
    expected = coefficient(lambda t: 3 * t)

    if map_func is qutip.solver.parallel.loky_pmap:
        loky = pytest.importorskip("loky")
        otherwise_never_used += " + 0"

    coeffs = map_func(coefficient, [otherwise_never_used] * 10)

    for coeff in coeffs:
        assert isinstance(coeff, Coefficient)
        _assert_eq_over_interval(coeff, expected)

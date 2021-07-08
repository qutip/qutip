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
from qutip import *
import numpy as np
from numpy.testing import assert_allclose
from functools import partial
from types import FunctionType, BuiltinFunctionType

from qutip.core import data as _data

# prepare coefficient
class Pseudo_qevo:
    # Mimic QobjEvo on __call__
    # and can return parameter to create the equivalent QobjEvo
    # for each coefficient type
    def __init__(self, cte, qobj, func, string, args):
        self.cte = cte
        self.qobj = qobj
        self.func = func
        self.str = string
        self.args = args

    def array(self):
        tlist = np.linspace(0, 10, 10001)
        coeff = self.func(tlist, self.args)
        return ([self.cte, [self.qobj, coeff]], {}, tlist)

    def logarray(self):
        tlist = np.logspace(-3, 1, 10001)
        coeff = self.func(tlist, self.args)
        return ([self.cte, [self.qobj, coeff]], {}, tlist)

    def spline(self):
        tlist = np.linspace(0, 10, 10001)
        coeff = Cubic_Spline(tlist[0], tlist[-1], self.func(tlist, self.args))
        return ([self.cte, [self.qobj, coeff]], )

    def func_coeff(self):
        return ([self.cte, [self.qobj, self.func]], self.args)

    def string(self):
        return ([self.cte, [self.qobj, self.str]], self.args)

    def func_call(self):
        return (self.__call__, self.args)

    def __call__(self, t, args={}):
        args = args or self.args
        return self.cte + self.qobj * self.func(t, args)

    def __getitem__(self, which):
        return getattr(self, which)()

N = 3
args = {'w1': 1, "w2": 2}
TESTTIMES = np.linspace(0.001, 1.0, 10)


def _real(t, args):
    return np.sin(t*args['w1'])


def _cplx(t, args):
    return np.exp(1j*t*args['w2'])


real_qevo = Pseudo_qevo(
    rand_stochastic(N).to(data.CSR),
    rand_stochastic(N).to(data.CSR),
    _real, "sin(t*w1)", args)

herm_qevo = Pseudo_qevo(
    rand_herm(N).to(data.Dense),
    rand_herm(N).to(data.Dense),
    _real, "sin(t*w1)", args)

cplx_qevo = Pseudo_qevo(
    rand_stochastic(N).to(data.Dense),
    rand_stochastic(N).to(data.CSR) + rand_stochastic(N).to(data.CSR) * 1j,
    _cplx, "exp(1j*t*w2)", args)


@pytest.fixture(params=['func_coeff', 'string', 'spline',
                        'array', 'logarray', 'func_call'])
def coeff_type(request):
    # all available QobjEvo types
    return request.param


@pytest.fixture(params=[
    pytest.param(real_qevo, id="real"),
    pytest.param(herm_qevo, id="hermitian"),
    pytest.param(cplx_qevo, id="complex"),
])
def pseudo_qevo(request):
    return request.param


@pytest.fixture
def all_qevo(pseudo_qevo, coeff_type):
    return QobjEvo(*pseudo_qevo[coeff_type])


@pytest.fixture
def other_qevo(all_qevo):
    return all_qevo


def _assert_qobjevo_equivalent(obj1, obj2, tol=1e-8):
    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj1(t), obj2(t), tol)


def _assert_qobj_almost_eq(obj1, obj2, tol=1e-10):
    assert _data.iszero((obj1 - obj2).data, tol)


def _assert_qobjevo_different(obj1, obj2):
    assert any(obj1(t) != obj2(t) for t in np.random.rand(10) * .9 + 0.05)


def _div(a, b):
    return a / b


def test_call(pseudo_qevo, coeff_type):
    # test creation of QobjEvo and call
    qevo = QobjEvo(*pseudo_qevo[coeff_type])
    assert isinstance(qevo(0), Qobj)
    assert qevo.isoper
    assert not qevo.isconstant
    assert not qevo.issuper
    _assert_qobjevo_equivalent(pseudo_qevo, qevo)

@pytest.mark.parametrize('coeff_type',
                         ['func_coeff', 'string',
                          'spline', 'array', 'logarray'])
def test_product_coeff(pseudo_qevo, coeff_type):
    # test creation of QobjEvo with Qobj * Coefficient
    # Skip pure func: QobjEvo(f(t, args) -> Qobj)
    base = pseudo_qevo[coeff_type]
    cte, [qobj, coeff] = base[0]
    args = base[1] if len(base) >= 2 else {}
    tlist = base[2] if len(base) >= 3 else None
    coeff = coefficient(coeff, args=args, tlist=tlist)
    created = cte + qobj * coeff
    _assert_qobjevo_equivalent(pseudo_qevo, created)

def test_copy(all_qevo):
    qevo = all_qevo
    copy = qevo.copy()
    _assert_qobjevo_equivalent(copy, qevo)
    assert copy is not qevo

@pytest.mark.parametrize('bin_op', [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a @ b, id="matmul"),
    pytest.param(lambda a, b: a & b, id="tensor"),
])
def test_binopt(all_qevo, other_qevo, bin_op):
    "QobjEvo arithmetic"
    obj1 = all_qevo
    obj2 = other_qevo
    for t in TESTTIMES:
        as_qevo = bin_op(obj1, obj2)(t)
        as_qobj = bin_op(obj1(t), obj2(t))
        _assert_qobj_almost_eq(as_qevo, as_qobj)

@pytest.mark.parametrize('bin_op', [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a @ b, id="matmul"),
    pytest.param(lambda a, b: a & b, id="tensor"),
])
def test_binopt_qobj(all_qevo, bin_op):
    "QobjEvo arithmetic"
    obj = all_qevo
    qobj = rand_herm(N)
    for t in TESTTIMES:
        as_qevo = bin_op(obj, qobj)(t)
        as_qobj = bin_op(obj(t), qobj)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        as_qevo = bin_op(qobj, obj)(t)
        as_qobj = bin_op(qobj, obj(t))
        _assert_qobj_almost_eq(as_qevo, as_qobj)

@pytest.mark.parametrize('bin_op', [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(_div, id="div"),
])
def test_binopt_scalar(all_qevo, bin_op):
    "QobjEvo arithmetic"
    obj = all_qevo
    scalar = 0.5 + 1j
    for t in TESTTIMES:
        as_qevo = bin_op(obj, scalar)(t)
        as_qobj = bin_op(obj(t), scalar)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        if bin_op is not _div:
            as_qevo = bin_op(scalar, obj)(t)
            as_qobj = bin_op(scalar, obj(t))
            _assert_qobj_almost_eq(as_qevo, as_qobj)

def binop_coeff(all_qevo):
    obj = all_qevo
    coeff = coeffient("t")
    created = obj * coeff_t
    for t in TESTTIMES:
        _assert_qobj_almost_eq(created(t), obj(t) * t)

@pytest.mark.parametrize('unary_op', [
    pytest.param(lambda a: a.conj(), id="conj"),
    pytest.param(lambda a: a.dag(), id="dag"),
    pytest.param(lambda a: a.trans(), id="trans"),
    pytest.param(lambda a: -a, id="neg"),
])
def test_unary(all_qevo, unary_op):
    "QobjEvo arithmetic"
    obj = all_qevo
    for t in TESTTIMES:
        as_qevo = unary_op(obj)(t)
        as_qobj = unary_op(obj(t))
        _assert_qobj_almost_eq(as_qevo, as_qobj)

@pytest.mark.parametrize('args_coeff_type',
                         ['func_coeff', 'string', 'func_call'])
def test_args(pseudo_qevo, args_coeff_type):
    obj = QobjEvo(*pseudo_qevo[args_coeff_type])
    args = {'w1': 3, "w2": 3}

    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj(t, args), pseudo_qevo(t, args))

    # Did it modify original args
    _assert_qobjevo_equivalent(obj, pseudo_qevo)

    obj.arguments(args)
    _assert_qobjevo_different(obj, pseudo_qevo)
    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj(t), pseudo_qevo(t, args))

def test_copy_side_effects(all_qevo):
    t = 0.2
    qevo = all_qevo
    copy = qevo.copy()
    before = qevo(t)
    # Ensure inplace modification of the copy do not affect the original
    copy *= 2
    copy += rand_herm(N)
    copy *= rand_herm(N)
    copy.arguments({'w1': 3, "w2": 3})
    after = qevo(t)
    _assert_qobj_almost_eq(before, after)

@pytest.mark.parametrize('coeff_type',
    ['func_coeff', 'string', 'spline', 'array', 'logarray']
)
def test_tidyup(all_qevo):
    "QobjEvo tidyup"
    obj = all_qevo
    obj *= 1e-12
    obj.tidyup(atol=1e-8)
    t = 0.2
    # check that the Qobj are cleaned
    assert_allclose(obj(t).full(), 0)

def test_QobjEvo_pickle(all_qevo):
    "QobjEvo pickle"
    # used in parallel_map
    import pickle
    obj = all_qevo
    pickled = pickle.dumps(obj, -1)
    recreated = pickle.loads(pickled)
    _assert_qobjevo_equivalent(recreated, obj)

def test_shift(all_qevo):
    dt = 0.2
    obj = all_qevo
    shited = obj._insert_time_shift(dt)
    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj(t + dt), shited(t))

def test_mul_vec(all_qevo):
    "QobjEvo matmul ket"
    vec = Qobj(np.arange(N)*.5+.5j)
    op = all_qevo
    for t in TESTTIMES:
        assert_allclose((op(t) @ vec).full(),
                        op.matmul(t, vec).full(), atol=1e-14)

def test_matmul(all_qevo):
    "QobjEvo matmul oper"
    mat = np.random.rand(N, N) + 1 + 1j * np.random.rand(N, N)
    matDense = Qobj(mat).to(_data.Dense)
    matF = Qobj(np.asfortranarray(mat)).to(_data.Dense)
    matCSR = Qobj(mat).to(_data.CSR)
    op = all_qevo
    for t in TESTTIMES:
        Qo1 = op(t)
        assert_allclose((Qo1 @ mat).full(),
                        op.matmul(t, matF).full(), atol=1e-14)
        assert_allclose((Qo1 @ mat).full(),
                        op.matmul(t, matDense).full(), atol=1e-14)
        assert_allclose((Qo1 @ mat).full(),
                        op.matmul(t, matCSR).full(), atol=1e-14)

def test_expect_psi(all_qevo):
    "QobjEvo expect psi"
    vec = _data.dense.fast_from_numpy(np.arange(N)*.5 + .5j)
    qobj = Qobj(vec)
    op = all_qevo
    for t in TESTTIMES:
        Qo1 = op(t)
        assert_allclose(_data.expect(Qo1.data, vec), op.expect(t, qobj),
                        atol=1e-14)

def test_expect_rho(all_qevo):
    "QobjEvo expect rho"
    vec = _data.dense.fast_from_numpy(np.random.rand(N*N) + 1
                                      + 1j * np.random.rand(N*N))
    mat = _data.column_unstack_dense(vec, N)
    qobj = Qobj(mat)
    op = liouvillian(all_qevo)
    for t in TESTTIMES:
        Qo1 = op(t)
        assert abs(_data.expect_super(Qo1.data, vec)
                   - op.expect(t, qobj)) < 1e-14

@pytest.mark.parametrize('dtype',
[pytest.param(dtype, id=dtype.__name__)
     for dtype in core.data.to.dtypes])
def test_convert(all_qevo, dtype):
    "QobjEvo expect rho"
    op = all_qevo.to(dtype)
    assert isinstance(op(0.5).data, dtype)

def test_compress():
    "QobjEvo compress"
    obj = QobjEvo([[qeye(N), "t"], [qeye(N), "t"], [qeye(N), "t"]])
    before = obj.num_elements
    obj2 = obj.copy()
    obj2.compress()
    assert before >= obj2.num_elements
    _assert_qobjevo_equivalent(obj, obj2)


@pytest.mark.parametrize(['qobjdtype'],
    [pytest.param(dtype, id=dtype.__name__)
     for dtype in core.data.to.dtypes])
@pytest.mark.parametrize(['statedtype'],
    [pytest.param(dtype, id=dtype.__name__)
     for dtype in core.data.to.dtypes])
def test_layer_support(qobjdtype, statedtype):
    N = 10
    qevo = QobjEvo(rand_herm(N).to(qobjdtype))
    state_dense = rand_ket(N).to(core.data.Dense)
    state = state_dense.to(statedtype).data
    state_dense = state_dense.data
    exp_any = qevo.expect_data(0, state)
    exp_dense = qevo.expect_data(0, state_dense)
    assert_allclose(exp_any, exp_dense)
    mul_any = qevo.matmul_data(0, state).to_array()
    mul_dense = qevo.matmul_data(0, state_dense).to_array()
    assert_allclose(mul_any, mul_dense)


def test_QobjEvo_step_coeff():
    "QobjEvo step interpolation"
    coeff1 = np.random.rand(6)
    coeff2 = np.random.rand(6) + np.random.rand(6) * 1.j
    # uniform t
    tlist = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
                      tlist=tlist, step_interpolation=True)
    assert qobjevo(2.0)[0,0] == coeff1[0]
    assert qobjevo(7.0)[0,0] == coeff1[5]
    assert qobjevo(5.0001)[0,0] == coeff1[3]
    assert qobjevo(3.9999)[0,0] == coeff1[1]

    assert qobjevo(2.0)[0,1] == coeff2[0]
    assert qobjevo(7.0)[0,1] == coeff2[5]
    assert qobjevo(5.0001)[0,1] == coeff2[3]
    assert qobjevo(3.9999)[0,1] == coeff2[1]

    # non-uniform t
    tlist = np.array([1, 2, 4, 5, 6, 8], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
        tlist=tlist, step_interpolation=True)
    assert qobjevo(1.0)[0,0] == coeff1[0]
    assert qobjevo(8.0)[0,0] == coeff1[5]
    assert qobjevo(3.9999)[0,0] == coeff1[1]
    assert qobjevo(4.23)[0,0] == coeff1[2]
    assert qobjevo(1.23)[0,0] == coeff1[0]

    assert qobjevo(1.0)[0,1] == coeff2[0]
    assert qobjevo(8.0)[0,1] == coeff2[5]
    assert qobjevo(6.7)[0,1] == coeff2[4]
    assert qobjevo(7.9999)[0,1] == coeff2[4]
    assert qobjevo(3.9999)[0,1] == coeff2[1]

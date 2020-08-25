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
from qutip import Cubic_Spline
from qutip.core.cy.cqobjevo import CQobjEvo

from qutip.core import data as _data


class Base_coeff:
    def __init__(self, func, string):
        self.func = func
        self.string = string

    def array(self, tlist, args):
        return self.func(tlist, args)

    def spline(self, tlist, args):
        return Cubic_Spline(tlist[0], tlist[-1], self.func(tlist, args))

    def __call__(self, t, args):
        return self.func(t, args)


def _real(t, args):
    return np.sin(t*args['w1'])


def _cplx(t, args):
    return np.exp(1j*t*args['w2'])


@pytest.fixture(params=[
    pytest.param("cte", id="cte"),
    pytest.param("func", id="func"),
    pytest.param("string", id="string"),
    pytest.param("spline", id="spline"),
    pytest.param("array", id="array"),
    pytest.param("array_log", id="array_log"),
    pytest.param("with_args", id="with_args"),
    pytest.param("no_args", id="no_args"),
    pytest.param("mixed_tlist", id="mixed_tlist"),
])
def form(request):
    return request.param


@pytest.fixture(params=[
    pytest.param("func", id="func"),
    pytest.param("string", id="string"),
    pytest.param("with_args", id="with_args"),
])
def args_form(request):
    return request.param


@pytest.fixture(params=[
    pytest.param("cte", id="cte"),
    pytest.param("func", id="func"),
    pytest.param("string", id="string"),
    pytest.param("spline", id="spline"),
    pytest.param("array", id="array"),
    pytest.param("array_log", id="array_log"),
    pytest.param("with_args", id="with_args"),
    pytest.param("no_args", id="no_args"),
    pytest.param("mixed_tlist", id="mixed_tlist"),
])
def extra_form(request):
    return request.param


def _assert_qobjevo_equivalent(obj1, obj2, tol=1e-8):
    for _ in range(10):
        t = np.random.rand() * .9 + 0.05
        _assert_qobj_almost_eq(obj1(t), obj2(t), tol)


def _assert_qobj_almost_eq(obj1, obj2, tol=1e-10):
    assert _data.iszero((obj1 - obj2).data, tol)


def _assert_qobjevo_different(obj1, obj2):
    assert any(obj1(t) != obj2(t) for t in np.random.rand(10) * .9 + 0.05)


def _add(a, b):
    return a + b

def _sub(a, b):
    return a - b

def _mul(a, b):
    return a * b

def _div(a, b):
    return a / b

def _matmul(a, b):
    return a @ b

def _tensor(a, b):
    return a & b

def _conj(a):
    return a.conj()

def _dag(a):
    return a.dag()

def _trans(a):
    return a.trans()

def _neg(a):
    return -a

def _cdc(a):
    if isinstance(a, Qobj):
        return a.dag() * a
    return a._cdc()


class TestQobjevo:
    N = 4
    tlist = np.linspace(0, 10, 10001)
    tlistlog = np.logspace(-3, 1, 10001)
    args = {'w1': 1, "w2": 2}
    cte_qobj = rand_herm(N)
    real_qobj = Qobj(np.random.rand(N, N))
    cplx_qobj = rand_herm(N)
    real_coeff = Base_coeff(_real, "sin(t*w1)")
    cplx_coeff = Base_coeff(_cplx, "exp(1j*t*w2)")
    qobjevos = {"qobj": rand_herm(N), "scalar":np.random.rand()}
    cplx_forms = ['func', 'string', 'spline', 'array', 'array_log']
    mixed_forms = ['with_args', 'no_args', 'mixed_tlist']

    def _rand_t(self):
        return np.random.rand() * 0.9 + 0.05

    @pytest.mark.parametrize('tlist', [tlist], ids=['test1'])
    def test_param(self, tlist):
        assert tlist is self.tlist

    @pytest.mark.parametrize(['form', 'base', 'kwargs'],
        [pytest.param('cte', cte_qobj, {}, id='cte'),
         pytest.param('func', [cte_qobj,
                               [cplx_qobj, cplx_coeff.func]],
                      {'args':args}, id='func'),
         pytest.param('string', [cte_qobj,
                                 [cplx_qobj, cplx_coeff.string]],
                      {'args':args}, id='string'),
         pytest.param('spline', [[cplx_qobj, cplx_coeff.spline(tlist, args)],
                                 cte_qobj],
                      {}, id='spline'),
         pytest.param('array', [cte_qobj,
                                [cplx_qobj, cplx_coeff.array(tlist, args)]],
                      {'tlist':tlist}, id='array'),
         pytest.param('array_log', [cte_qobj,
                                [cplx_qobj, cplx_coeff.array(tlistlog, args)]],
                      {'tlist':tlistlog}, id='array_log'),
         pytest.param('with_args', [cte_qobj,
                                    [cplx_qobj, cplx_coeff.func],
                                    [real_qobj, real_coeff.string]],
                      {'args':args}, id='with_args'),
         pytest.param('no_args', [cte_qobj,
                                  [cplx_qobj, cplx_coeff.spline(tlist, args)],
                                  [real_qobj, real_coeff.array(tlist, args)]],
                      {'tlist':tlist}, id='no_args'),
         pytest.param('mixed_tlist', [cte_qobj,
                        [cplx_qobj, coefficient(
                                        cplx_coeff.array(tlist, args),
                                        tlist=tlist
                                    )],
                        [real_qobj, coefficient(
                                        real_coeff.array(tlistlog, args),
                                        tlist=tlistlog
                                    )]],
                      {}, id='mixed_tlist'),
    ])
    def test_creation(self, form, base, kwargs):
        obj = QobjEvo(base, **kwargs)
        self.qobjevos[form] = obj
        assert isinstance(obj.compiled_qobjevo, CQobjEvo)

    def test_call(self, form):
        t = self._rand_t()
        assert isinstance(self.qobjevos[form](t), Qobj)

        for _ in range(10):
            t = self._rand_t()
            if form == "cte":
                _assert_qobj_almost_eq(self.qobjevos[form](t), self.cte_qobj)
            elif form in self.cplx_forms:
                expected = (self.cte_qobj +
                            self.cplx_coeff(t, self.args) * self.cplx_qobj)
                _assert_qobj_almost_eq(self.qobjevos[form](t), expected)
            elif form in self.mixed_forms:
                expected = (self.cte_qobj +
                            self.cplx_coeff(t, self.args) * self.cplx_qobj +
                            self.real_coeff(t, self.args) * self.real_qobj)
                _assert_qobj_almost_eq(self.qobjevos[form](t), expected)

    def test_product_coeff(self):
        coeff_1 = coefficient("1")
        coeff_t = coefficient("t")
        created = self.cte_qobj * coeff_1
        _assert_qobjevo_equivalent(self.qobjevos['cte'], created)

        created = created * coeff_t
        for _ in range(10):
            t = self._rand_t()
            expected = (self.cte_qobj * t)
            _assert_qobj_almost_eq(created(t), expected)

    def test_copy(self, form):
        copy = self.qobjevos[form].copy()
        _assert_qobjevo_equivalent(copy, self.qobjevos[form])

        assert copy is not self.qobjevos[form]
        t = self._rand_t()
        before = self.qobjevos[form](t)
        copy.cte *= 2
        for op in copy.ops:
            op.coeff = coefficient("0")
        after = self.qobjevos[form](t)
        _assert_qobj_almost_eq(before, after)

    def test_args(self, args_form):
        copy = self.qobjevos[args_form].copy()
        args = {'w1': 3, "w2": 3}

        for _ in range(10):
            t = self._rand_t()
            if args_form in self.cplx_forms:
                expected = (self.cte_qobj +
                            self.cplx_coeff(t, args) * self.cplx_qobj)
                _assert_qobj_almost_eq(copy(t, args=args), expected)
            elif args_form in self.mixed_forms:
                expected = (self.cte_qobj +
                            self.cplx_coeff(t, args) * self.cplx_qobj +
                            self.real_coeff(t, args) * self.real_qobj)
                _assert_qobj_almost_eq(copy(t, args=args), expected)

        _assert_qobjevo_equivalent(copy, self.qobjevos[args_form])

        copy.arguments(args)
        _assert_qobjevo_different(copy, self.qobjevos[args_form])
        for _ in range(10):
            t = self._rand_t()
            if args_form in self.cplx_forms:
                expected = (self.cte_qobj +
                            self.cplx_coeff(t, args) * self.cplx_qobj)
                _assert_qobj_almost_eq(copy(t), expected)
            elif args_form in self.mixed_forms:
                expected = (self.cte_qobj +
                            self.cplx_coeff(t, args) * self.cplx_qobj +
                            self.real_coeff(t, args) * self.real_qobj)
                _assert_qobj_almost_eq(copy(t), expected)

    @pytest.mark.parametrize('bin_op', [_add, _sub, _mul, _matmul, _tensor],
                             ids=['add', 'sub', 'mul', 'matmul', 'tensor'])
    def test_binopt(self, form, extra_form, bin_op):
        "QobjEvo arithmetic"
        t = self._rand_t()
        obj1 = self.qobjevos[form]
        obj1_t = self.qobjevos[form](t)
        obj2 = self.qobjevos[extra_form]
        obj2_t = self.qobjevos[extra_form](t)

        as_qevo = bin_op(obj1, obj2)(t)
        as_qobj = bin_op(obj1_t, obj2_t)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

    @pytest.mark.parametrize('bin_op', [_add, _sub, _mul, _matmul, _tensor],
                             ids=['add', 'sub', 'mul', 'matmul', 'tensor'])
    def test_binopt_qobj(self, form, bin_op):
        "QobjEvo arithmetic"
        t = self._rand_t()
        obj = self.qobjevos[form]
        obj_t = self.qobjevos[form](t)
        qobj = rand_herm(self.N)

        as_qevo = bin_op(obj, qobj)(t)
        as_qobj = bin_op(obj_t, qobj)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        as_qevo = bin_op(qobj, obj)(t)
        as_qobj = bin_op(qobj, obj_t)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

    @pytest.mark.parametrize('bin_op', [_add, _sub, _mul, _div],
                             ids=['add', 'sub', 'mul', 'div'])
    def test_binopt_scalar(self, form, bin_op):
        "QobjEvo arithmetic"
        t = self._rand_t()
        obj = self.qobjevos[form]
        obj_t = self.qobjevos[form](t)
        scalar = np.random.rand() + 1j

        as_qevo = bin_op(obj, scalar)(t)
        as_qobj = bin_op(obj_t, scalar)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        if bin_op is not _div:
            as_qevo = bin_op(scalar, obj)(t)
            as_qobj = bin_op(scalar, obj_t)
            _assert_qobj_almost_eq(as_qevo, as_qobj)

    @pytest.mark.parametrize('unary_op', [_neg, _dag, _conj, _trans, _cdc],
                             ids=['neg', 'dag', 'conj', 'trans', '_cdc'])
    def test_unary(self, form, extra_form, unary_op):
        "QobjEvo arithmetic"
        t = self._rand_t()
        obj = self.qobjevos[form]
        obj_t = self.qobjevos[form](t)

        as_qevo = unary_op(obj)(t)
        as_qobj = unary_op(obj_t)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

    def test_tidyup(self):
        "QobjEvo tidyup"
        obj = self.qobjevos["no_args"]
        obj *= 1e-10 * np.random.random()
        obj.tidyup(atol=1e-8)
        t = self._rand_t()
        # check that the Qobj are cleaned
        assert_allclose(obj(t).full(), 0, atol=1e-14)

    def test_QobjEvo_pickle(self, form):
        "QobjEvo pickle"
        # used in parallel_map
        import pickle
        pickled = pickle.dumps(self.qobjevos[form], -1)
        recreated = pickle.loads(pickled)
        _assert_qobjevo_equivalent(recreated, self.qobjevos[form])

    @pytest.mark.parametrize('superop', [
        pytest.param(lindblad_dissipator, id='lindblad'),
        pytest.param(partial(lindblad_dissipator, chi=0.5), id='lindblad_chi'),
        pytest.param(sprepost, id='sprepost'),
        pytest.param(lambda O1, O2: liouvillian(O1, [O2]), id='liouvillian')
    ])
    def test_superoperator_qobj(self, form, superop):
        t = self._rand_t()
        obj1 = self.qobjevos[form]
        obj1_t = self.qobjevos[form](t)
        qobj = rand_herm(self.N)

        as_qevo = superop(obj1, qobj)(t)
        as_qobj = superop(obj1_t, qobj)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        as_qevo = superop(qobj, obj1)(t)
        as_qobj = superop(qobj, obj1_t)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

    @pytest.mark.parametrize('superop', [
        pytest.param(lindblad_dissipator, id='lindblad'),
        pytest.param(partial(lindblad_dissipator, chi=0.5), id='lindblad_chi'),
        pytest.param(sprepost, id='sprepost'),
        pytest.param(lambda O1, O2: liouvillian(O1, [O2]), id='liouvillian')
    ])
    def test_superoperator(self, form, extra_form, superop):
        t = self._rand_t()
        obj1 = self.qobjevos[form]
        obj1_t = self.qobjevos[form](t)
        obj2 = self.qobjevos[extra_form]
        obj2_t = self.qobjevos[extra_form](t)

        as_qevo = superop(obj1, obj2)(t)
        as_qobj = superop(obj1_t, obj2_t)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

    def test_shift(self, form):
        t = self._rand_t() / 2
        dt = self._rand_t() / 2
        obj = self.qobjevos[form]
        shifted = obj._shift()
        _assert_qobj_almost_eq(obj(t + dt), shifted(t, args={"_t0": dt}))

    def test_QobjEvo_apply(self):
        "QobjEvo apply"
        obj = self.qobjevos["no_args"]
        transposed = obj.apply(_trans)
        _assert_qobjevo_equivalent(transposed, obj.trans())

    def test_mul_vec(self, form):
        "QobjEvo mul_numpy"
        t = self._rand_t()
        N = self.N
        vec = np.arange(N)*.5+.5j
        op = self.qobjevos[form]
        assert_allclose(op(t).full() @ vec, op.mul(t, vec), atol=1e-14)

    def test_mul_mat(self, form):
        "QobjEvo mul_mat"
        t = self._rand_t()
        N = self.N
        mat = np.random.rand(N, N) + 1 + 1j * np.random.rand(N, N)
        matF = np.asfortranarray(mat)
        op = self.qobjevos[form]
        Qo1 = op(t)
        assert_allclose(Qo1.full() @ mat, op.mul(t, mat), atol=1e-14)
        assert_allclose(Qo1.full() @ matF, op.mul(t, matF), atol=1e-14)

    def test_expect_psi(self, form):
        "QobjEvo expect psi"
        t = self._rand_t()
        N = self.N
        vec = _data.dense.fast_from_numpy(np.arange(N)*.5 + .5j)
        op = self.qobjevos[form]
        Qo1 = op(t)
        assert_allclose(_data.expect(Qo1.data, vec), op.expect(t, vec),
                        atol=1e-14)

    def test_expect_rho(self, form):
        "QobjEvo expect rho"
        t = self._rand_t()
        N = self.N
        vec = _data.dense.fast_from_numpy(np.random.rand(N*N) + 1
                                          + 1j * np.random.rand(N*N))
        mat = _data.column_unstack_dense(vec, N)
        qobj = Qobj(mat.to_array())
        op = liouvillian(self.qobjevos[form])
        Qo1 = op(t)
        assert abs(_data.expect_super(Qo1.data, vec)
                   - op.expect(t, vec, 0)) < 1e-14
        assert abs(_data.expect_super(Qo1.data, vec)
                   - op.expect(t, mat, 0)) < 1e-14
        assert abs(_data.expect_super(Qo1.data, vec)
                   - op.expect(t, qobj, 0)) < 1e-14

    def test_compress(self):
        "QobjEvo compress"
        obj = self.qobjevos["no_args"]
        obj2 = (obj + obj) / 2
        before = obj2.num_obj
        obj2.compress()
        assert before >= obj2.num_obj
        _assert_qobjevo_equivalent(obj, obj2)


def test_QobjEvo_step_coeff():
    "QobjEvo step interpolation"
    coeff1 = np.random.rand(6)
    coeff2 = np.random.rand(6) + np.random.rand(6) * 1.j
    # uniform t
    tlist = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
                      tlist=tlist, args={"_step_func_coeff": True})
    assert qobjevo.ops[0].coeff(2.0) == coeff1[0]
    assert qobjevo.ops[0].coeff(7.0) == coeff1[5]
    assert qobjevo.ops[0].coeff(5.0001) == coeff1[3]
    assert qobjevo.ops[0].coeff(3.9999) == coeff1[1]

    assert qobjevo.ops[1].coeff(2.0) == coeff2[0]
    assert qobjevo.ops[1].coeff(7.0) == coeff2[5]
    assert qobjevo.ops[1].coeff(5.0001) == coeff2[3]
    assert qobjevo.ops[1].coeff(3.9999) == coeff2[1]

    # non-uniform t
    tlist = np.array([1, 2, 4, 5, 6, 8], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
        tlist=tlist, args={"_step_func_coeff":True})
    assert qobjevo.ops[0].coeff(1.0) == coeff1[0]
    assert qobjevo.ops[0].coeff(8.0) == coeff1[5]
    assert qobjevo.ops[0].coeff(3.9999) == coeff1[1]
    assert qobjevo.ops[0].coeff(4.23) == coeff1[2]
    assert qobjevo.ops[0].coeff(1.23) == coeff1[0]

    assert qobjevo.ops[1].coeff(1.0) == coeff2[0]
    assert qobjevo.ops[1].coeff(8.0) == coeff2[5]
    assert qobjevo.ops[1].coeff(6.7) == coeff2[4]
    assert qobjevo.ops[1].coeff(7.9999) == coeff2[4]
    assert qobjevo.ops[1].coeff(3.9999) == coeff2[1]


@pytest.mark.skipif(True, reason="Now returning Coefficient, to adapt/remove")
def test_QobjEvo_to_list():
    "QobjEvo to_list"
    td_as_list_1 = _random_QobjEvo((5,5), [0,2,3], tlist=np.linspace(0,1,100))
    td_as_list_2 = _random_QobjEvo((5,5), [1,0,0], tlist=np.linspace(0,1,100))
    args={"w1":1, "w2":2}
    td_obj_1 = QobjEvo(td_as_list_1, args=args, tlist=np.linspace(0,1,100))
    td_obj_2 = QobjEvo(td_as_list_2, args=args, tlist=np.linspace(0,1,100))
    td_as_list_back = (td_obj_1 + td_obj_2).to_list()

    for part in td_as_list_back:
        if isinstance(part, Qobj):
            assert td_as_list_1[0] + td_as_list_2[0] == part
        elif isinstance(part[1], (FunctionType, BuiltinFunctionType, partial)):
            assert td_as_list_2[1][1] == part[1]
            assert td_as_list_2[1][0] == part[0]
        elif isinstance(part[1], str):
            assert td_as_list_1[1][1] == part[1]
            assert td_as_list_1[1][0] == part[0]
        elif isinstance(part[1], np.ndarray):
            assert (td_as_list_1[2][1] == part[1]).all()
            assert td_as_list_1[2][0] == part[0]
        else:
            assert False


# TODO remove when with state moved to solvers
def test_QobjEvo_with_state():
    "QobjEvo dynamics_args"
    def coeff_state(t, args):
        return np.mean(args["state_vec"]) * args["w"] * args["expect_op_0"]
    N = 5
    vec = np.arange(N)*.5+.5j
    t = np.random.random()
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    args = {"w": 5, "state_vec": None, "expect_op_0": 2*qeye(N)}

    td_data = QobjEvo([q1, [q2, coeff_state]], args=args, e_ops=[2*qeye(N)])
    q_at_t = q1 + np.mean(vec) * args["w"] * expect(2*qeye(N), Qobj(vec.T)) * q2
    # Check that the with_state call
    assert_allclose(td_data.mul(t, vec), (q_at_t * vec).full()[:, 0],
                    atol=1e-14)
    # Check that the with_state call compiled
    assert_allclose(td_data.mul(t, vec), (q_at_t * vec).full()[:, 0],
                    atol=1e-14)

    td_data = QobjEvo([q1, [q2, "state_vec[0] * cos(w*expect_op_0*t)"]],
                      args=args, e_ops=[2*qeye(N)])
    data_at_t = q1 + q2*vec[0]*np.cos(10 * t * expect(qeye(N), Qobj(vec.T)))
    # Check that the with_state call for str format
    assert_allclose(td_data.mul(t, vec), (data_at_t * vec).full()[:, 0],
                    atol=1e-14)
    # Check that the with_state call for str format and compiled
    assert_allclose(td_data.mul(t, vec), (data_at_t * vec).full()[:, 0],
                    atol=1e-14)

    args = {"state_mat": None, "state_vec": None, "state_qobj": None}
    mat = np.arange(N*N).reshape((N, N))

    def check_dyn_args(t, args):
        assert isinstance(args["state_qobj"], Qobj)
        assert isinstance(args["state_vec"], np.ndarray)
        assert isinstance(args["state_mat"], np.ndarray)

        assert len(args["state_vec"].shape) == 1
        assert len(args["state_mat"].shape) == 2

        assert (np.all(args["state_vec"]
                == args["state_qobj"].full().ravel("F")))
        assert np.all(args["state_vec"] == args["state_mat"].ravel("F"))
        assert np.all(args["state_mat"] == mat)
        return 1

    td_data = QobjEvo([q1, check_dyn_args], args=args)
    td_data.mul(0, mat)

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
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
                           run_module_suite, assert_allclose)
from functools import partial
from types import FunctionType, BuiltinFunctionType
from qutip import Cubic_Spline

from qutip.core import data as _data


@pytest.fixture(params=[
    pytest.param({'dense': 1}, id="dense"),
    pytest.param({'matched': 1}, id="matched"),
    pytest.param({'omp': 2}, id="omp"),
    pytest.param({'matched': 1, 'omp': 2}, id="matched,omp"),
])
def compile_kwargs(request):
    return request.param


def _f1(t, args):
    return np.sin(t*args['w1'])


def _f2(t, args):
    return np.cos(t*args['w2'])


def _f3(t, args):
    return np.exp(1j*t*args['w3'])


def _rand_cqobjevo(N=5):
    tlist = np.linspace(0, 10, 10001)
    tlistlog = np.logspace(-3, 1, 10001)
    O0, O1, O2 = rand_herm(N), rand_herm(N), rand_herm(N)
    cte = [QobjEvo([O0])]
    wargs = [QobjEvo([O0, [O1, _f1], [O2, _f2]], args={"w1": 1, "w2": 2}),
             QobjEvo([O0, [O1, "sin(w1*t)"], [O2, "cos(w2*t)"]],
                     args={"w1": 1, "w2": 2})]
    nargs = [QobjEvo([O0, [O1, np.sin(tlist)], [O2, np.cos(2*tlist)]],
                     tlist=tlist),
             QobjEvo([O0, [O1, np.sin(tlistlog)], [O2, np.cos(2*tlistlog)]],
                     tlist=tlistlog),
             QobjEvo([O0, [O1, Cubic_Spline(0, 10, np.sin(tlist))],
                      [O2, Cubic_Spline(0, 10, np.cos(2*tlist))]])]
    cqobjevos = cte + wargs + nargs
    base_qobjs = [O0, O1, O2]
    return cqobjevos, base_qobjs


def _sp_eq(sp1, sp2):
    return not np.any(np.abs((sp1 - sp2).as_scipy().data) > 1e-4)


def _random_QobjEvo(shape=(1,1), ops=[0,0,0], cte=True, tlist=None):
    """Create a list to make a QobjEvo with up to 3 coefficients"""
    if tlist is None:
        tlist = np.linspace(0,1,301)
    Qobj_list = []
    if cte:
        Qobj_list.append(Qobj(np.random.random(shape) + \
                              1j*np.random.random(shape)))
    coeff =  [[_f1, "sin(w1*t)",    np.sin(tlist),
                    Cubic_Spline(0,1,np.sin(tlist))],
              [_f2, "cos(w2*t)",    np.cos(tlist),
                    Cubic_Spline(0,1,np.cos(tlist))],
              [_f3, "exp(w3*t*1j)", np.exp(tlist*1j),
                    Cubic_Spline(0,1,np.exp(tlist*1j))]]
    for i, form in enumerate(ops):
        if form:
            Qobj_list.append([Qobj(np.random.random(shape)), coeff[i][form-1]])
    return Qobj_list


def _assert_qobj_almost_eq(obj1, obj2, tol=1e-10):
    diff_data = (obj1 - obj2).tidyup(tol).data
    assert _data.csr.nnz(diff_data) == 0


@pytest.mark.parametrize('use_cython', [True, False],
                         ids=['cython', 'no cython'])
def test_QobjEvo_call(use_cython):
    "QobjEvo call"
    N = 5
    t = np.random.rand()+1
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    O0, O1, O2 = base_qobjs
    O_target1 = O0 + np.sin(t)*O1 + np.cos(2*t)*O2

    # Check the constant flag
    assert cqobjevos[0].const
    # Check that the call return the Qobj
    assert cqobjevos[0](t) == O0
    # Check that the call for the data return the data
    assert _sp_eq(cqobjevos[0](t, data=True), O0.data)

    for op in cqobjevos[1:]:
        assert _sp_eq(op(t, data=True), O_target1.data)
        _assert_qobj_almost_eq(op(t), O_target1)
        op.use_cython = use_cython
        op.compile()
        assert _sp_eq(op(t, data=True), O_target1.data)
        _assert_qobj_almost_eq(op(t), O_target1)


@pytest.mark.slow
def test_QobjEvo_call_full(compile_kwargs):
    "QobjEvo call"
    N = 5
    t = np.random.rand()+1
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    O0, O1, O2 = base_qobjs
    O_target1 = O0 + np.sin(t)*O1 + np.cos(2*t)*O2
    # Check the constant flag
    assert cqobjevos[0].const
    # Check that the call return the Qobj
    assert cqobjevos[0](t) == O0
    # Check that the call for the data return the data
    assert _sp_eq(cqobjevos[0](t, data=True), O0.data)
    for op in cqobjevos[1:]:
        op.compile(**compile_kwargs)
        assert _sp_eq(op(t, data=True), O_target1.data)
        _assert_qobj_almost_eq(op(t), O_target1)


def test_QobjEvo_call_args_uncompiled():
    "QobjEvo with_args"
    N = 5
    t = np.random.rand()+1
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    O0, O1, O2 = base_qobjs
    O_target1 = O0 + np.sin(t)*O1 + np.cos(2*t)*O2
    O_target2 = O0 + np.sin(t)*O1 + np.cos(4*t)*O2
    for op in cqobjevos[1:3]:
        _assert_qobj_almost_eq(op(t, args={"w2": 4}), O_target2)
        op.arguments({"w2": 4})
        _assert_qobj_almost_eq(op(t), O_target2)
        op.arguments({"w2": 2})
        _assert_qobj_almost_eq(op(t), O_target1)


@pytest.mark.slow
def test_QobjEvo_call_args_compiled(compile_kwargs):
    N = 5
    t = np.random.rand()+1
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    O0, O1, O2 = base_qobjs
    O_target1 = O0 + np.sin(t)*O1 + np.cos(2*t)*O2
    O_target2 = O0 + np.sin(t)*O1 + np.cos(4*t)*O2
    for op in cqobjevos[1:3]:
        op.compile(**compile_kwargs)
        _assert_qobj_almost_eq(op(t, args={"w2": 4}), O_target2)
        op.arguments({"w2": 4})
        _assert_qobj_almost_eq(op(t), O_target2)
        op.arguments({"w2": 2})
        _assert_qobj_almost_eq(op(t), O_target1)


def test_QobjEvo_step_coeff():
    "QobjEvo step interpolation"
    coeff1 = np.random.rand(6)
    coeff2 = np.random.rand(6) + np.random.rand(6) * 1.j
    # uniform t
    tlist = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
                      tlist=tlist, args={"_step_func_coeff": True})
    assert qobjevo.ops[0].get_coeff(2.0) == coeff1[0]
    assert qobjevo.ops[0].get_coeff(7.0) == coeff1[5]
    assert qobjevo.ops[0].get_coeff(5.0001) == coeff1[3]
    assert qobjevo.ops[0].get_coeff(3.9999) == coeff1[1]

    assert qobjevo.ops[1].get_coeff(2.0) == coeff2[0]
    assert qobjevo.ops[1].get_coeff(7.0) == coeff2[5]
    assert qobjevo.ops[1].get_coeff(5.0001) == coeff2[3]
    assert qobjevo.ops[1].get_coeff(3.9999) == coeff2[1]

    qobjevo.compile()
    assert (qobjevo.coeff_get(2.0) == [coeff1[0], coeff2[0]]).all()
    assert (qobjevo.coeff_get(7.0) == [coeff1[5], coeff2[5]]).all()
    assert (qobjevo.coeff_get(4.0001) == [coeff1[2], coeff2[2]]).all()
    assert (qobjevo.coeff_get(3.9999) == [coeff1[1], coeff2[1]]).all()

    # non-uniform t
    tlist = np.array([1, 2, 4, 5, 6, 8], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
        tlist=tlist, args={"_step_func_coeff":True})
    assert qobjevo.ops[0].get_coeff(1.0) == coeff1[0]
    assert qobjevo.ops[0].get_coeff(8.0) == coeff1[5]
    assert qobjevo.ops[0].get_coeff(3.9999) == coeff1[1]
    assert qobjevo.ops[0].get_coeff(4.23) == coeff1[2]
    assert qobjevo.ops[0].get_coeff(1.23) == coeff1[0]

    assert qobjevo.ops[1].get_coeff(1.0) == coeff2[0]
    assert qobjevo.ops[1].get_coeff(8.0) == coeff2[5]
    assert qobjevo.ops[1].get_coeff(6.7) == coeff2[4]
    assert qobjevo.ops[1].get_coeff(7.9999) == coeff2[4]
    assert qobjevo.ops[1].get_coeff(3.9999) == coeff2[1]

    qobjevo.compile()
    assert (qobjevo.coeff_get(1.0) == [coeff1[0], coeff2[0]]).all()
    assert (qobjevo.coeff_get(3.999) == [coeff1[1], coeff2[1]]).all()
    assert (qobjevo.coeff_get(6.3) == [coeff1[4], coeff2[4]]).all()
    assert (qobjevo.coeff_get(1.0001) == [coeff1[0], coeff2[0]]).all()


def test_QobjEvo_copy():
    "QobjEvo copy"
    tlist = np.linspace(0,1,300)
    td_obj_1 = QobjEvo(_random_QobjEvo((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    t = np.random.random()
    td_obj_copy = td_obj_1.copy()
    #Check that the copy is independent
    assert td_obj_1 is not td_obj_copy
    #Check that the copy has the same data
    assert td_obj_1(t) == td_obj_copy(t)
    td_obj_copy = QobjEvo(td_obj_1)
    #Check that the copy is independent
    assert td_obj_1 is not td_obj_copy
    #Check that the copy has the same data
    assert td_obj_1(t) == td_obj_copy(t)


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


def test_QobjEvo_math_arithmetic():
    "QobjEvo arithmetic"
    N = 5
    t = np.random.rand()+1
    cqobjevos1, base_qobjs1 = _rand_cqobjevo(N)
    cqobjevos2, base_qobjs2 = _rand_cqobjevo(N)
    cte = cqobjevos2[0]
    O1 = base_qobjs2[0]

    for op, op_2 in zip(cqobjevos1, cqobjevos2):
        _assert_qobj_almost_eq(op(t)*-1, (-op)(t) )

        _assert_qobj_almost_eq(op(t) +O1, (op +O1)(t))
        _assert_qobj_almost_eq(op(t) +cte(t), (op +cte)(t))
        _assert_qobj_almost_eq(op(t) +op_2(t), (op +op_2)(t))
        opp = op.copy()
        opp += O1
        _assert_qobj_almost_eq(op(t) +O1, opp(t))
        opp = op.copy()
        opp += op_2
        _assert_qobj_almost_eq(op(t) +op_2(t), opp(t))

        _assert_qobj_almost_eq(op(t) -O1, (op -O1)(t))
        _assert_qobj_almost_eq(op(t) -cte(t), (op -cte)(t))
        _assert_qobj_almost_eq(O1 -op(t), (O1 -op)(t))
        _assert_qobj_almost_eq(cte(t) -op(t), (cte -op)(t))
        _assert_qobj_almost_eq(op(t) -op_2(t), (op -op_2)(t))
        opp = op.copy()
        opp -= O1
        _assert_qobj_almost_eq(op(t) -O1, opp(t))
        opp = op.copy()
        opp -= op_2
        _assert_qobj_almost_eq(op(t) -op_2(t), opp(t))

        _assert_qobj_almost_eq(op(t) * O1, (op * O1)(t))
        _assert_qobj_almost_eq(O1 * op(t), (O1 * op)(t))
        _assert_qobj_almost_eq(2 * op(t), (2 * op)(t))
        _assert_qobj_almost_eq(op(t) * cte(t), (op * cte)(t))
        _assert_qobj_almost_eq(cte(t) * op(t), (cte * op)(t))
        _assert_qobj_almost_eq(op(t) * op_2(t), (op * op_2)(t))
        _assert_qobj_almost_eq(op_2(t) * op(t), (op_2 * op)(t))
        opp = op.copy()
        opp *= 2
        _assert_qobj_almost_eq(2 * op(t), opp(t))
        opp = op.copy()
        opp *= O1
        _assert_qobj_almost_eq(op(t) * O1, opp(t))
        opp = op.copy()
        opp *= op_2
        _assert_qobj_almost_eq(op(t) * op_2(t), opp(t))

        _assert_qobj_almost_eq(op(t)/2, (op/2)(t))
        opp = op.copy()
        opp /= 2
        _assert_qobj_almost_eq(op(t)/2, opp(t))


def test_QobjEvo_unitary():
    "QobjEvo trans, dag, conj, _cdc"
    N = 5
    t = np.random.rand()+1
    cqobjevos, base_qobjs = _rand_cqobjevo(N)

    for op in cqobjevos:
        _assert_qobj_almost_eq((op.trans())(t), op(t).trans())
        _assert_qobj_almost_eq((op.dag())(t), op(t).dag())
        _assert_qobj_almost_eq((op.conj())(t), op(t).conj())
        _assert_qobj_almost_eq((op._cdc())(t), op(t).dag()*op(t))


def test_QobjEvo_tidyup():
    "QobjEvo tidyup"
    tlist = np.linspace(0,1,300)
    args={"w1":1}
    td_obj = QobjEvo(_random_QobjEvo((5,5), [1,0,0], tlist=tlist),
                     args=args, tlist=tlist)
    td_obj *= 1e-10 * np.random.random()
    td_obj.tidyup(atol=1e-8)
    t = np.random.random()
    # check that the Qobj are cleaned
    assert_equal(np.max(td_obj(t, data=True).to_array()), 0.)


def test_QobjEvo_compress():
    "QobjEvo compress"
    tlist = np.linspace(0, 1, 300)
    td_obj_1 = QobjEvo(_random_QobjEvo((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_2 = (td_obj_1 + td_obj_1)/2.
    t = np.random.random()
    td_obj_2.compress()
    # check that the number of part is decreased
    assert_equal(len(td_obj_2.to_list()), 4)
    # check that data is still valid
    _assert_qobj_almost_eq(td_obj_2(t), td_obj_1(t))


def test_QobjEvo_shift():
    """QobjEvo _shift time"""
    tlist = np.linspace(0, 1, 300)
    td_obj_1 = QobjEvo(_random_QobjEvo((1,1), [0,0], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_1s = td_obj_1.copy()
    td_obj_1s._shift()
    td_obj_2 = QobjEvo(_random_QobjEvo((1,1), [1,1], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_2s = td_obj_2.copy()
    td_obj_2s._shift()

    _assert_qobj_almost_eq(td_obj_1(0),td_obj_1s(1, args={"_t0":-1}))
    _assert_qobj_almost_eq(td_obj_2(0),td_obj_2s(1, args={"_t0":-1}))
    td_obj_1s.arguments({"_t0":-1})
    td_obj_2s.arguments({"_t0":-1})
    _assert_qobj_almost_eq(td_obj_1(0),td_obj_1s(1))
    _assert_qobj_almost_eq(td_obj_2(0),td_obj_2s(1))
    td_obj_1s.compile()
    td_obj_2s.compile()
    _assert_qobj_almost_eq(td_obj_1(0),td_obj_1s(1))
    _assert_qobj_almost_eq(td_obj_2(0),td_obj_2s(1))
    _assert_qobj_almost_eq(td_obj_1(0),td_obj_1s(2, args={"_t0":-2}))
    _assert_qobj_almost_eq(td_obj_2(0),td_obj_2s(2, args={"_t0":-2}))
    td_obj_1s.arguments({"_t0":-2})
    td_obj_2s.arguments({"_t0":-2})
    _assert_qobj_almost_eq(td_obj_1(0),td_obj_1s(2))
    _assert_qobj_almost_eq(td_obj_2(0),td_obj_2s(2))


def test_QobjEvo_apply():
    "QobjEvo apply"
    def multiply(qobj,b,factor = 3.):
        return qobj*b*factor
    tlist = np.linspace(0, 1, 300)
    td_obj = QobjEvo(_random_QobjEvo((5,5), [1,2,3], tlist=tlist),
                     args={"w1":1, "w2":2}, tlist=tlist)
    t = np.random.random()
    # check that the number of part is decreased
    assert_equal(td_obj.apply(multiply,2)(t) == td_obj(t)*6, True)
    # check that data is still valid
    assert_equal(td_obj.apply(multiply,2,factor=2)(t) == td_obj(t)*4, True)


def test_QobjEvo_apply_decorator():
    "QobjEvo apply_decorator"
    def rescale_time_and_scale(f_original, time_scale, factor=1.):
        def f(t, *args, **kwargs):
            return f_original(time_scale*t, *args, **kwargs)*factor
        return f

    tlist = np.linspace(0, 1, 501)
    td_obj = QobjEvo(_random_QobjEvo((5,5), [1,2,3], tlist=tlist, cte=False),
                     args={"w1":1, "w2":2}, tlist=tlist)
    t = 0.10 + np.random.random() * 0.80
    # cubicspline interpolation can be less precise
    # at the limit of the time range.
    td_obj_scaled = td_obj.apply_decorator(rescale_time_and_scale,2)
    # check that the decorated took effect mixed
    assert_equal(td_obj_scaled(t) == td_obj(2*t), True)
    for op in td_obj_scaled.ops:
        assert_equal(op.type, "func")

    def square_f(f_original):
        def f(t, *args, **kwargs):
            return f_original(t, *args, **kwargs)**2
        return f
    td_list = _random_QobjEvo((5,5), [2, 0, 0], tlist=tlist, cte=False)
    td_obj_str = QobjEvo(td_list, args={"w1": 1, "w2": 2}, tlist=tlist)
    td_obj_str_2 = td_obj_str.apply_decorator(square_f, str_mod=["(", ")**2"])
    _assert_qobj_almost_eq(td_obj_str_2(t), td_list[0][0] * np.sin(t)**2)
    assert_equal(td_obj_str_2.ops[0].type, "string")

    td_list = _random_QobjEvo((5,5), [3, 0, 0], tlist=tlist, cte=False)
    td_obj_array = QobjEvo(td_list, tlist=tlist)
    td_obj_array_2 = td_obj_array.apply_decorator(square_f, inplace_np=True)
    _assert_qobj_almost_eq(td_obj_array_2(t),
                           td_list[0][0] * np.sin(t)**2, tol=3e-7)
    assert_equal(td_obj_array_2.ops[0].type, "array")


def test_QobjEvo_mul_vec():
    "QobjEvo mul_vec"
    N = 5
    t = np.random.rand()+1
    vec = np.arange(N)*.5+.5j
    cqobjevos, _ = _rand_cqobjevo(N)

    for op in cqobjevos:
        assert_allclose(op(t, data=1).to_array() @ vec,
                        op.mul_vec(t, vec))
        op.compile()
        assert_allclose(op(t, data=1).to_array() @ vec,
                        op.mul_vec(t, vec))


@pytest.mark.slow
def test_QobjEvo_mul_vec_full(compile_kwargs):
    "QobjEvo mul_vec"
    N = 5
    t = np.random.rand()+1
    vec = np.arange(N)*.5+.5j
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    for op in cqobjevos:
        op.compile(**compile_kwargs)
        assert_allclose(op(t, data=1).to_array() @ vec,
                        op.mul_vec(t, vec))


def test_QobjEvo_mul_mat():
    "QobjEvo mul_mat"
    N = 5
    t = np.random.rand()+1
    mat = np.random.rand(N, N)+1 + 1j*np.random.rand(N,N)
    matF = np.asfortranarray(mat)
    cqobjevos, base_qobjs = _rand_cqobjevo(N)

    for op in cqobjevos:
        Qo1 = op(t)
        assert_allclose(Qo1.full() @ mat, op.mul_mat(t, mat))
        assert_allclose(Qo1.full() @ matF, op.mul_mat(t, matF))
        op.compile()
        assert_allclose(Qo1.full() @ mat, op.mul_mat(t, mat))
        assert_allclose(Qo1.full() @ matF, op.mul_mat(t, matF))


@pytest.mark.slow
def test_QobjEvo_mul_mat_full(compile_kwargs):
    "QobjEvo mul_mat"
    N = 5
    t = np.random.rand()+1
    mat = np.random.rand(N, N) + 1 + 1j*np.random.rand(N, N)
    matF = np.asfortranarray(mat)
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    for op in cqobjevos:
        Qo1 = op(t)
        op.compile(**compile_kwargs)
        assert_allclose(Qo1.full() @ mat, op.mul_mat(t, mat))
        assert_allclose(Qo1.full() @ matF, op.mul_mat(t, matF))


def test_QobjEvo_expect_psi():
    "QobjEvo expect psi"
    N = 5
    t = np.random.rand()+1
    vec = _data.dense.fast_from_numpy(np.arange(N)*.5 + .5j)
    cqobjevos, base_qobjs = _rand_cqobjevo(N)
    _expect = _data.expect_csr_dense

    for op in cqobjevos:
        Qo1 = op(t)
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec)) < 1e-10
        op.compile()
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec)) < 1e-10


@pytest.mark.slow
def test_QobjEvo_expect_psi_full(compile_kwargs):
    "QobjEvo expect psi"
    N = 5
    t = np.random.rand()+1
    vec = _data.dense.fast_from_numpy(np.arange(N)*.5 + .5j)
    cqobjevos, _ = _rand_cqobjevo(N)
    for op in cqobjevos:
        Qo1 = op(t)
        op.compile(**compile_kwargs)
        assert abs(_data.expect_csr_dense(Qo1.data, vec)
                   - op.expect(t, vec)) < 1e-10


def test_QobjEvo_expect_rho():
    "QobjEvo expect rho"
    N = 5
    t = np.random.rand()+1
    vec = _data.dense.fast_from_numpy(np.random.rand(N*N)
                                      + 1
                                      + 1j*np.random.rand(N*N))
    mat = _data.column_unstack_dense(vec, N)
    qobj = Qobj(mat.to_array())
    _expect = _data.expect_super_csr_dense
    cqobjevos, _ = _rand_cqobjevo(N)

    for op_ in cqobjevos:
        op = liouvillian(op_)
        Qo1 = op(t)
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, mat, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, qobj, 0)) < 1e-14
        op.compile()
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, mat, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, qobj, 0)) < 1e-14

    tlist = np.linspace(0,1,300)
    args = {"w1": 1, "w2": 2, "w3": 3}
    td_obj_sa = QobjEvo(_random_QobjEvo((3,3), [0,3,2], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_m = QobjEvo(_random_QobjEvo((3,3), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)
    t = np.random.random()
    td_obj_sa = td_obj_sa.apply(spre)
    td_obj_m = td_obj_m.apply(spre)
    rho = _data.dense.fast_from_numpy(np.arange(3*3)*0.25+.25j)
    td_obj_sac = td_obj_sa.copy()
    td_obj_sac.compile()
    v1 = td_obj_sa.expect(t, rho, 0)
    v2 = td_obj_sac.expect(t, rho, 0)
    v3 = _expect(td_obj_sa(t, data=True), rho)
    # check not compiled rhs const
    assert_allclose(v1, v3, rtol=1e-6)
    # check compiled rhs
    assert_allclose(v3, v2, rtol=1e-6)

    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    v1 = td_obj_m.expect(t, rho, 1)
    v2 = td_obj_mc.expect(t, rho, 1)
    v3 = _expect(td_obj_m(t, data=True), rho).real
    # check not compiled rhs func
    assert_allclose(v1, v3, rtol=1e-6)
    # check compiled rhs func
    assert_allclose(v3, v2, rtol=1e-6)


@pytest.mark.slow
def test_QobjEvo_expect_rho_full():
    "QobjEvo expect rho"
    N = 5
    t = np.random.rand()+1
    vec = _data.dense.fast_from_numpy(np.random.rand(N*N)
                                      + 1
                                      + 1j*np.random.rand(N*N))
    mat = _data.column_unstack_dense(vec, N)
    qobj = Qobj(mat.to_array())
    _expect = _data.expect_super_csr_dense
    cqobjevos, base_qobjs = _rand_cqobjevo(N)

    for op_ in cqobjevos:
        op = liouvillian(op_)
        Qo1 = op(t)
        op.compile(dense=1)
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, mat, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, qobj, 0)) < 1e-14
        op.compiled = ""
        op.compile(matched=1)
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, mat, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, qobj, 0)) < 1e-14
        op.compiled = ""
        op.compile(omp=2)
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, mat, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, qobj, 0)) < 1e-14
        op.compiled = ""
        op.compile(matched=1,omp=2)
        assert abs(_expect(Qo1.data, vec) - op.expect(t, vec, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, mat, 0)) < 1e-14
        assert abs(_expect(Qo1.data, vec) - op.expect(t, qobj, 0)) < 1e-14

    tlist = np.linspace(0,1,300)
    args={"w1":1, "w2":2, "w3":3}
    td_obj_sa = QobjEvo(_random_QobjEvo((3,3), [0,3,2], tlist=tlist),
                        args=args, tlist=tlist)
    td_obj_m = QobjEvo(_random_QobjEvo((3,3), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)
    t = np.random.random()
    td_obj_sa = td_obj_sa.apply(spre)
    td_obj_m = td_obj_m.apply(spre)
    rho = _data.dense.fast_from_numpy(np.arange(3*3)*0.25+.25j)
    td_obj_sac = td_obj_sa.copy()
    td_obj_sac.compile()
    v1 = td_obj_sa.expect(t, rho, 0)
    v2 = td_obj_sac.expect(t, rho, 0)
    v3 = _expect(td_obj_sa(t, data=True), rho)
    # check not compiled rhs const
    assert_allclose(v1, v3, rtol=1e-6)
    # check compiled rhs
    assert_allclose(v3, v2, rtol=1e-6)

    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    v1 = td_obj_m.expect(t, rho, 1)
    v2 = td_obj_mc.expect(t, rho, 1)
    v3 = _expect(td_obj_m(t, data=True), rho).real
    # check not compiled rhs func
    assert_allclose(v1, v3, rtol=1e-6)
    # check compiled rhs func
    assert_allclose(v3, v2, rtol=1e-6)


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
    assert_allclose(td_data.mul_vec(t, vec), (q_at_t * vec).full()[:, 0])
    td_data.compile()
    # Check that the with_state call compiled
    assert_allclose(td_data.mul_vec(t, vec), (q_at_t * vec).full()[:, 0])

    td_data = QobjEvo([q1, [q2, "state_vec[0] * cos(w*expect_op_0*t)"]],
                      args=args, e_ops=[2*qeye(N)])
    data_at_t = q1 + q2*vec[0]*np.cos(10 * t * expect(qeye(N), Qobj(vec.T)))
    # Check that the with_state call for str format
    assert_allclose(td_data.mul_vec(t, vec), (data_at_t * vec).full()[:, 0])
    td_data.compile()
    # Check that the with_state call for str format and compiled
    assert_allclose(td_data.mul_vec(t, vec), (data_at_t * vec).full()[:, 0])

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
    td_data.mul_mat(0, mat)


def test_QobjEvo_pickle():
    "QobjEvo pickle"
    #used in parallel_map
    import pickle
    tlist = np.linspace(0,1,300)
    args={"w1":1, "w2":2}
    t = np.random.random()

    td_obj_c = QobjEvo(_random_QobjEvo((5,5), [0,0,0]))
    td_obj_c.compile()
    pickled = pickle.dumps(td_obj_c)
    td_pick = pickle.loads(pickled)
    # Check for const case
    assert_equal(td_obj_c(t) == td_pick(t), True)

    td_obj_sa = QobjEvo(_random_QobjEvo((5,5), [2,3,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_sa.compile()
    td_obj_m = QobjEvo(_random_QobjEvo((5,5), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)

    pickled = pickle.dumps(td_obj_sa, -1)
    td_pick = pickle.loads(pickled)
    # Check for cython compiled coeff
    assert_equal(td_obj_sa(t) == td_pick(t), True)

    pickled = pickle.dumps(td_obj_m, -1)
    td_pick = pickle.loads(pickled)
    # Check for not compiled mix
    assert_equal(td_obj_m(t) == td_pick(t), True)
    td_obj_m.compile()
    pickled = pickle.dumps(td_obj_m, -1)
    td_pick = pickle.loads(pickled)
    # Check for ct_cqobjevo
    assert_equal(td_obj_m(t) == td_pick(t), True)


def test_QobjEvo_superoperator():
    "QobjEvo superoperator"
    cqobjevos1, _ = _rand_cqobjevo(3)
    cqobjevos2, _ = _rand_cqobjevo(3)
    t = np.random.rand()+1
    for op1, op2 in zip(cqobjevos1, cqobjevos2):
        Q1 = op1(t)
        Q2 = op2(t)
        _assert_qobj_almost_eq(lindblad_dissipator(Q1, Q2, chi=0.5),
                               lindblad_dissipator(op1, op2, chi=0.5)(t))
        _assert_qobj_almost_eq(sprepost(Q1, Q2),
                               sprepost(op1, op2)(t))
        _assert_qobj_almost_eq(liouvillian(Q1, [Q2]),
                               liouvillian(op1, [op2])(t))

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
from qutip import *
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
                            run_module_suite, assert_allclose)
from functools import partial
from types import FunctionType, BuiltinFunctionType
from qutip.interpolate import Cubic_Spline
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, spmv)

def _f1(t,args):
    return np.sin(t*args['w1'])

def _f2(t,args):
    return np.cos(t*args['w2'])

def _f3(t,args):
    return np.exp(1j*t*args['w3'])


def _rand_td_qobj(N=5):
    tlist=np.linspace(0,10,10001)
    tlistlog=np.logspace(-3,1,10001)
    O0, O1, O2 = rand_herm(N), rand_herm(N), rand_herm(N)

    cte = [td_Qobj([O0])]

    wargs = [td_Qobj([O0,[O1,_f1],[O2,_f2]], args={"w1":1,"w2":2}),
             td_Qobj([O0,[O1,"sin(w1*t)"],[O2,"cos(w2*t)"]],
                     args={"w1":1,"w2":2})]

    nargs = [td_Qobj([O0,[O1,np.sin(tlist)],[O2,np.cos(2*tlist)]],tlist=tlist),
             td_Qobj([O0,[O1,np.sin(tlistlog)],[O2,np.cos(2*tlistlog)]],
                     tlist=tlistlog),
             td_Qobj([O0,[O1, Cubic_Spline(0,10,np.sin(tlist))  ],
                         [O2, Cubic_Spline(0,10,np.cos(2*tlist))]])]
    td_qobjs = cte + wargs + nargs
    base_qobjs = [O0, O1, O2]
    return td_qobjs, base_qobjs


def _sp_eq(sp1, sp2):
    return not np.any(np.abs( (sp1 -sp2).data)> 1e-4)


def _random_td_Qobj(shape=(1,1), ops=[0,0,0], cte=True, tlist=None):
    """Create a list to make a td_Qobj with up to 3 coefficients"""
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
    for i,form in enumerate(ops):
        if form:
            Qobj_list.append([Qobj(np.random.random(shape)),coeff[i][form-1]])
    return Qobj_list


def test_td_Qobj_call():
    "td_Qobj call"
    N = 5
    t = np.random.rand()+1
    td_qobjs, base_qobjs = _rand_td_qobj(N)
    O0, O1, O2 = base_qobjs
    O_target1 = O0+np.sin(t)*O1+np.cos(2*t)*O2

    # Check the constant flag
    assert_equal(td_qobjs[0].const, True)
    # Check that the call return the Qobj
    assert_equal(td_qobjs[0](t) == O0, True)
    # Check that the call for the data return the data
    assert_equal(_sp_eq(td_qobjs[0](t, data=True), O0.data), True)

    for op in td_qobjs[1:]:
        assert_equal(_sp_eq(op(t, data=1) , O_target1.data), True)
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile()
        assert_equal(_sp_eq(op(t, data=1) , O_target1.data), True)
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(dense=1)
        assert_equal(_sp_eq(op(t, data=1) , O_target1.data), True)
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(matched=1)
        assert_equal(_sp_eq(op(t, data=1) , O_target1.data), True)
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(omp=2)
        assert_equal(_sp_eq(op(t, data=1) , O_target1.data), True)
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(matched=1, omp=2)
        assert_equal(_sp_eq(op(t, data=1) , O_target1.data), True)
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)


def test_td_Qobj_call_args():
    "td_Qobj with_args"
    N = 5
    t = np.random.rand()+1
    td_qobjs, base_qobjs = _rand_td_qobj(N)
    O0, O1, O2 = base_qobjs
    O_target1 = O0+np.sin(t)*O1+np.cos(2*t)*O2
    O_target2 = O0+np.sin(t)*O1+np.cos(4*t)*O2

    for op in td_qobjs[1:3]:
        assert_equal(len((op.with_args(t,{"w2":4})
                          -O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":4})
        assert_equal(len((op(t) - O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":2})
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile()
        assert_equal(len((op.with_args(t,{"w2":4})
                          -O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":4})
        assert_equal(len((op(t) - O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":2})
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(dense=1)
        assert_equal(len((op.with_args(t,{"w2":4})
                          -O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":4})
        assert_equal(len((op(t) - O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":2})
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(matched=1)
        assert_equal(len((op.with_args(t,{"w2":4})
                          -O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":4})
        assert_equal(len((op(t) - O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":2})
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(omp=2)
        assert_equal(len((op.with_args(t,{"w2":4})
                          -O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":4})
        assert_equal(len((op(t) - O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":2})
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)

        op.compile(matched=1, omp=2)
        assert_equal(len((op.with_args(t,{"w2":4})
                          -O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":4})
        assert_equal(len((op(t) - O_target2).tidyup(1e-10).data.data),0)
        op.arguments({"w2":2})
        assert_equal(len((op(t) - O_target1).tidyup(1e-10).data.data),0)


def test_td_Qobj_copy():
    "td_Qobj copy"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    t = np.random.random()
    td_obj_copy = td_obj_1.copy()
    #Check that the copy is independent
    assert_equal(td_obj_1 is td_obj_copy, False)
    #Check that the copy has the same data
    assert_equal(td_obj_1(t) == td_obj_copy(t), True)
    td_obj_copy = td_Qobj(td_obj_1)
    #Check that the copy is independent
    assert_equal(td_obj_1 is td_obj_copy, False)
    #Check that the copy has the same data
    assert_equal(td_obj_1(t) == td_obj_copy(t), True)


def test_td_Qobj_to_list():
    "td_Qobj to_list"
    td_as_list_1 = _random_td_Qobj((5,5), [0,2,3], tlist=np.linspace(0,1,100))
    td_as_list_2 = _random_td_Qobj((5,5), [1,0,0], tlist=np.linspace(0,1,100))
    args={"w1":1, "w2":2}
    td_obj_1 = td_Qobj(td_as_list_1, args=args, tlist=np.linspace(0,1,100))
    td_obj_2 = td_Qobj(td_as_list_2, args=args, tlist=np.linspace(0,1,100))
    td_as_list_back = (td_obj_1 + td_obj_2).to_list()

    all_match = True
    for part in td_as_list_back:
        if isinstance(part, Qobj):
            all_match = all_match and td_as_list_1[0] + td_as_list_2[0] == part
        elif isinstance(part[1], (FunctionType, BuiltinFunctionType, partial)):
            all_match = all_match and td_as_list_2[1][1] == part[1]
            all_match = all_match and td_as_list_2[1][0] == part[0]
        elif isinstance(part[1], str):
            all_match = all_match and td_as_list_1[1][1] == part[1]
            all_match = all_match and td_as_list_1[1][0] == part[0]
        elif isinstance(part[1], np.ndarray):
            all_match = all_match and (td_as_list_1[2][1] == part[1]).all()
            all_match = all_match and td_as_list_1[2][0] == part[0]
        else:
            all_match = False
    # Check that the list get the object back
    assert_equal(all_match, True)


def test_td_Qobj_math_arithmetic():
    "td_Qobj arithmetic"
    N = 5
    t = np.random.rand()+1
    td_qobjs1, base_qobjs1 = _rand_td_qobj(N)
    td_qobjs2, base_qobjs2 = _rand_td_qobj(N)
    cte = td_qobjs2[0]
    O1 = base_qobjs2[0]

    for op, op_2 in zip(td_qobjs1, td_qobjs2):
        assert_equal(op(t)*-1, (-op)(t))

        assert_equal(op(t) +O1, (op +O1)(t))
        assert_equal(op(t) +cte(t), (op +cte)(t))
        assert_equal(op(t) +op_2(t), (op +op_2)(t))
        opp = op.copy()
        opp += O1
        assert_equal(op(t) +O1, opp(t))
        opp = op.copy()
        opp += op_2
        assert_equal(op(t) +op_2(t), opp(t))

        assert_equal(op(t) -O1, (op -O1)(t))
        assert_equal(op(t) -cte(t), (op -cte)(t))
        assert_equal(O1 -op(t), (O1 -op)(t))
        assert_equal(cte(t) -op(t), (cte -op)(t))
        assert_equal(op(t) -op_2(t), (op -op_2)(t))
        opp = op.copy()
        opp -= O1
        assert_equal(op(t) -O1, opp(t))
        opp = op.copy()
        opp -= op_2
        assert_equal(op(t) -op_2(t), opp(t))

        assert_equal(op(t) * O1, (op * O1)(t))
        assert_equal(O1 * op(t), (O1 * op)(t))
        assert_equal(2 * op(t), (2 * op)(t))
        assert_equal(op(t) * cte(t), (op * cte)(t))
        assert_equal(cte(t) * op(t), (cte * op)(t))
        assert_equal(op(t) * op_2(t), (op * op_2)(t))
        assert_equal(op_2(t) * op(t), (op_2 * op)(t))
        opp = op.copy()
        opp *= 2
        assert_equal(2 * op(t), opp(t))
        opp = op.copy()
        opp *= O1
        assert_equal(op(t) * O1, opp(t))
        opp = op.copy()
        opp *= op_2
        assert_equal(op(t) * op_2(t), opp(t))

        assert_equal(op(t)/2, (op/2)(t))
        opp = op.copy()
        opp /= 2
        assert_equal(op(t)/2, opp(t))


def test_td_Qobj_unitary():
    "td_Qobj trans, dag, conj, norm"
    N = 5
    t = np.random.rand()+1
    td_qobjs, base_qobjs = _rand_td_qobj(N)

    for op in td_qobjs:
        assert_equal((op.trans())(t), op(t).trans())
        assert_equal((op.dag())(t), op(t).dag())
        assert_equal((op.conj())(t), op(t).conj())
        assert_equal((op.norm())(t), op(t).dag()*op(t))


def test_td_Qobj_tidyup():
    "td_Qobj tidyup"
    tlist = np.linspace(0,1,300)
    args={"w1":1}
    td_obj = td_Qobj(_random_td_Qobj((5,5), [1,0,0], tlist=tlist),
                     args=args, tlist=tlist)
    td_obj *= 1e-10 * np.random.random()
    td_obj.tidyup(atol=1e-8)
    t = np.random.random()
    # check that the Qobj are cleaned
    assert_equal(np.max(td_obj(t, data=True)), 0.)


def test_td_Qobj_compress():
    "td_Qobj compress"
    tlist = np.linspace(0, 1, 300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_2 = (td_obj_1 + td_obj_1)/2.
    t = np.random.random()
    td_obj_2.compress()
    # check that the number of part is decreased
    assert_equal(len(td_obj_2.to_list()), 4)
    # check that data is still valid
    assert_equal(td_obj_2(t), td_obj_1(t))


def test_td_Qobj_apply():
    "td_Qobj apply"
    def multiply(qobj,b,factor = 3.):
        return qobj*b*factor
    tlist = np.linspace(0, 1, 300)
    td_obj = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                     args={"w1":1, "w2":2}, tlist=tlist)
    t = np.random.random()
    # check that the number of part is decreased
    assert_equal(td_obj.apply(multiply,2)(t) == td_obj(t)*6, True)
    # check that data is still valid
    assert_equal(td_obj.apply(multiply,2,factor=2)(t) == td_obj(t)*4, True)


def test_td_Qobj_apply_decorator():
    "td_Qobj apply_decorator"
    def rescale_time_and_scale(f_original, time_scale, factor=1.):
        def f(t, *args, **kwargs):
            return f_original(time_scale*t, *args, **kwargs)*factor
        return f

    tlist = np.linspace(0, 1, 300)
    td_obj = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist, cte=False),
                     args={"w1":1, "w2":2}, tlist=tlist)
    t = np.random.random() / 2.
    td_obj_scaled = td_obj.apply_decorator(rescale_time_and_scale,2)
    # check that the decorated took effect mixed
    assert_equal(td_obj_scaled(t) == td_obj(2*t), True)
    def square_f(f_original):
        def f(t, *args, **kwargs):
            return f_original(t, *args, **kwargs)**2
        return f

    td_list = _random_td_Qobj((5,5), [2,0,0], tlist=tlist, cte=False)
    td_obj_str = td_Qobj(td_list, args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_str_2 = td_obj_str.apply_decorator(square_f, str_mod=["(",")**2"])
    #Check that it can only be compiled to cython
    assert_equal(td_obj_str_2(t) == td_list[0][0] * np.sin(t)**2, True)

    td_list = _random_td_Qobj((5,5), [3,0,0], tlist=tlist, cte=False)
    td_obj_array = td_Qobj(td_list, tlist=tlist)
    td_obj_array_2 = td_obj_array.apply_decorator(square_f, inplace_np=True)
    #Check that it can only be compiled to cython
    assert_allclose(np.array(td_obj_array_2(t, data=True).todense()),
                    np.array((td_list[0][0] * np.sin(t)**2).data.todense()),
                    rtol=3e-7)


def test_td_Qobj_mul_vec():
    "td_Qobj mul_vec"
    N = 5
    t = np.random.rand()+1
    vec = np.arange(N)*.5+.5j
    td_qobjs, base_qobjs = _rand_td_qobj(N)

    for op in td_qobjs:
        assert_allclose(spmv(op(t,data=1), vec), op.mul_vec(t, vec))
        op.compile()
        assert_allclose(spmv(op(t,data=1), vec), op.mul_vec(t, vec))
        op.compile(dense=1)
        assert_allclose(spmv(op(t,data=1), vec), op.mul_vec(t, vec))
        op.compile(matched=1)
        assert_allclose(spmv(op(t,data=1), vec), op.mul_vec(t, vec))
        op.compile(omp=2)
        assert_allclose(spmv(op(t,data=1), vec), op.mul_vec(t, vec))
        op.compile(matched=1,omp=2)
        assert_allclose(spmv(op(t,data=1), vec), op.mul_vec(t, vec))


def test_td_Qobj_mul_mat():
    "td_Qobj mul_mat"
    N = 5
    t = np.random.rand()+1
    mat = np.random.rand(N,N)+1 + 1j*np.random.rand(N,N)
    matF = np.asfortranarray(mat)
    matV = mat2vec(mat).flatten()
    td_qobjs, base_qobjs = _rand_td_qobj(N)

    for op in td_qobjs:
        Qo1 = op(t)
        assert_allclose(Qo1.data * mat, op.mul_mat(t,mat))
        assert_allclose(Qo1.data * matF, op.mul_mat(t,matF))
        op.compile()
        assert_allclose(Qo1.data * mat, op.mul_mat(t,mat))
        assert_allclose(Qo1.data * matF, op.mul_mat(t,matF))
        assert_allclose(mat2vec(Qo1.data * mat).flatten(),
                        op.compiled_Qobj.ode_mul_mat_F_vec(t,matV))
        op.compile(dense=1)
        assert_allclose(Qo1.data * mat, op.mul_mat(t,mat))
        assert_allclose(Qo1.data * matF, op.mul_mat(t,matF))
        assert_allclose(mat2vec(Qo1.data * mat).flatten(),
                        op.compiled_Qobj.ode_mul_mat_F_vec(t,matV))
        op.compile(matched=1)
        assert_allclose(Qo1.data * mat, op.mul_mat(t,mat))
        assert_allclose(Qo1.data * matF, op.mul_mat(t,matF))
        assert_allclose(mat2vec(Qo1.data * mat).flatten(),
                        op.compiled_Qobj.ode_mul_mat_F_vec(t,matV))
        op.compile(omp=2)
        assert_allclose(Qo1.data * mat, op.mul_mat(t,mat))
        assert_allclose(Qo1.data * matF, op.mul_mat(t,matF))
        assert_allclose(mat2vec(Qo1.data * mat).flatten(),
                        op.compiled_Qobj.ode_mul_mat_F_vec(t,matV))
        op.compile(matched=1,omp=2)
        assert_allclose(Qo1.data * mat, op.mul_mat(t,mat))
        assert_allclose(Qo1.data * matF, op.mul_mat(t,matF))
        assert_allclose(mat2vec(Qo1.data * mat).flatten(),
                        op.compiled_Qobj.ode_mul_mat_F_vec(t,matV))


def test_td_Qobj_expect_psi():
    "td_Qobj expect psi"
    N = 5
    t = np.random.rand()+1
    vec = np.arange(N)*.5+.5j
    td_qobjs, base_qobjs = _rand_td_qobj(N)

    for op in td_qobjs:
        Qo1 = op(t)
        assert_allclose(cy_expect_psi(Qo1.data, vec, 0), op.expect(t,vec,0))
        op.compile()
        assert_allclose(cy_expect_psi(Qo1.data, vec, 0), op.expect(t,vec,0))
        op.compile(dense=1)
        assert_allclose(cy_expect_psi(Qo1.data, vec, 0), op.expect(t,vec,0))
        op.compile(matched=1)
        assert_allclose(cy_expect_psi(Qo1.data, vec, 0), op.expect(t,vec,0))
        op.compile(omp=2)
        assert_allclose(cy_expect_psi(Qo1.data, vec, 0), op.expect(t,vec,0))
        op.compile(matched=1,omp=2)
        assert_allclose(cy_expect_psi(Qo1.data, vec, 0), op.expect(t,vec,0))


def test_td_Qobj_expect_rho():
    "td_Qobj expect rho"
    N = 5
    t = np.random.rand()+1
    vec = np.random.rand(N*N)+1 + 1j*np.random.rand(N*N)
    td_qobjs, base_qobjs = _rand_td_qobj(N)

    for op_ in td_qobjs:
        op = liouvillian(op_)
        Qo1 = op(t)
        assert_allclose(cy_expect_rho_vec(Qo1.data, vec, 0),
                        op.expect(t,vec,0), atol=1e-14)
        op.compile()
        assert_allclose(cy_expect_rho_vec(Qo1.data, vec, 0),
                        op.expect(t,vec,0), atol=1e-14)
        op.compile(dense=1)
        assert_allclose(cy_expect_rho_vec(Qo1.data, vec, 0),
                        op.expect(t,vec,0), atol=1e-14)
        op.compile(matched=1)
        assert_allclose(cy_expect_rho_vec(Qo1.data, vec, 0),
                        op.expect(t,vec,0), atol=1e-14)
        op.compile(omp=2)
        assert_allclose(cy_expect_rho_vec(Qo1.data, vec, 0),
                        op.expect(t,vec,0), atol=1e-14)
        op.compile(matched=1,omp=2)
        assert_allclose(cy_expect_rho_vec(Qo1.data, vec, 0),
                        op.expect(t,vec,0), atol=1e-14)

    tlist = np.linspace(0,1,300)
    args={"w1":1, "w2":2, "w3":3}
    data1 = np.random.random((3,3))
    data2 = np.random.random((3,3))
    td_obj_sa = td_Qobj(_random_td_Qobj((3,3), [0,3,2], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_m = td_Qobj(_random_td_Qobj((3,3), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)
    t = np.random.random()
    td_obj_sa = td_obj_sa.apply(spre)
    td_obj_m = td_obj_m.apply(spre)
    rho = np.arange(3*3)*0.25+.25j
    td_obj_sac = td_obj_sa.copy()
    td_obj_sac.compile()
    v1 = td_obj_sa.expect(t, rho, 0)
    v2 = td_obj_sac.expect(t, rho, 0)
    v3 = cy_expect_rho_vec(td_obj_sa(t, data=True), rho, 0)
    # check not compiled rhs const
    assert_allclose(v1, v3, rtol=1e-6)
    # check compiled rhs
    assert_allclose(v3, v2, rtol=1e-6)

    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    v1 = td_obj_m.expect(t, rho, 1)
    v2 = td_obj_mc.expect(t, rho, 1)
    v3 = cy_expect_rho_vec(td_obj_m(t, data=True), rho, 1)
    # check not compiled rhs func
    assert_allclose(v1, v3, rtol=1e-6)
    # check compiled rhs func
    assert_allclose(v3, v2, rtol=1e-6)


def test_td_Qobj_with_state():
    "td_Qobj with_state"
    def coeff_state(t,psi,args):
        return np.mean(psi) * args["w1"]
    N = 5
    tlist = np.linspace(0,1,300)
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    data3 = np.random.random((N, N))
    data4 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)
    q4 = Qobj(data4)
    args={"w1":1}
    new_args={"w1":10}
    td_data = td_Qobj([q1,[q2,coeff_state]], args=args)
    vec = np.arange(N)*.5+.5j

    t = np.random.random()
    q_at_t = q1 + np.mean(vec) * args["w1"] * q2
    q_at_t_new = q1 + np.mean(vec) * new_args["w1"] * q2

    # Check that the with_state call
    assert_equal(td_data.with_state(t,vec) == q_at_t, True)
    assert_allclose(np.array(td_data.with_state(t, vec).data.todense()), np.array(q_at_t.data.todense()))
    # Check that the with_state call with custom args
    assert_equal(td_data.with_state(t, vec, new_args) == q_at_t_new, True)
    assert_allclose(np.array(td_data.with_state(t, vec, new_args).data.todense()), np.array(q_at_t_new.data.todense()))

    td_data.compile()
    # Check that the with_state call compiled
    assert_equal(td_data.with_state(t,vec) == q_at_t, True)
    assert_allclose(np.array(td_data.with_state(t, vec).data.todense()), np.array(q_at_t.data.todense()))
    # Check that the with_state call with custom args compiled
    assert_equal(td_data.with_state(t, vec, new_args) == q_at_t_new, True)
    assert_allclose(np.array(td_data.with_state(t, vec, new_args).data.todense()), np.array(q_at_t_new.data.todense()))

    args={"w1":1, "w2":2}
    new_args={"w2":10}
    t = np.random.random()
    td_data = td_Qobj([q1, [q2,coeff_state], [q3,"cos(w2*t)"],
                      [q4,np.exp(3*tlist)]], tlist=tlist, args=args)
    data_at_t = data1 + np.mean(vec) * args["w1"] * data2 +\
             np.cos(t*args["w2"]) * data3 + np.exp(3*t) * data4
    data_at_t_args = data1 + np.mean(vec) * args["w1"] * data2 +\
             np.cos(t*new_args["w2"]) * data3 + np.exp(3*t) * data4

    # Check that the with_state call for mixed format
    assert_allclose(np.array(td_data.with_state(t, vec, data=True).todense()), data_at_t)
    assert_allclose(np.array(td_data.with_state(t, vec).data.todense()), data_at_t)
    assert_allclose(np.array(td_data.with_state(t, vec, new_args,data=True).todense()), data_at_t_args)
    assert_allclose(np.array(td_data.with_state(t, vec, new_args).data.todense()), data_at_t_args)

    td_data.compile()
    # Check that the with_state call for mixed format and compiled
    assert_allclose(np.array(td_data.with_state(t, vec, data=True).todense()), data_at_t)
    assert_allclose(np.array(td_data.with_state(t, vec).data.todense()), data_at_t)
    assert_allclose(np.array(td_data.with_state(t, vec, new_args,data=True).todense()), data_at_t_args)
    assert_allclose(np.array(td_data.with_state(t, vec, new_args).data.todense()), data_at_t_args)


def test_td_Qobj_pickle():
    "td_Qobj pickle"
    #used in parallel_map
    import pickle
    tlist = np.linspace(0,1,300)
    args={"w1":1, "w2":2}
    t = np.random.random()

    td_obj_c = td_Qobj(_random_td_Qobj((5,5), [0,0,0]))
    td_obj_c.compile()
    pickled = pickle.dumps(td_obj_c)
    td_pick = pickle.loads(pickled)
    # Check for const case
    assert_equal(td_obj_c(t) == td_pick(t), True)

    td_obj_sa = td_Qobj(_random_td_Qobj((5,5), [2,3,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_sa.compile()
    td_obj_m = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
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
    # Check for ct_td_qobj
    assert_equal(td_obj_m(t) == td_pick(t), True)


def test_td_Qobj_superoperator():
    "td_Qobj superoperator"
    td_qobjs1, _ = _rand_td_qobj(3)
    td_qobjs2, _ = _rand_td_qobj(3)
    td_qobjs3, _ = _rand_td_qobj(3)
    t = np.random.rand()+1
    for op1, op2 in zip(td_qobjs1, td_qobjs2):
        Q1 = op1(t)
        Q2 = op2(t)
        assert_equal(lindblad_dissipator(Q1, Q2, chi=0.5),
                     lindblad_dissipator(op1, op2, chi=0.5)(t))
        assert_equal(sprepost(Q1, Q2), sprepost(op1, op2)(t))
        assert_equal( liouvillian(Q1, [Q2]), liouvillian(op1, [op2])(t))

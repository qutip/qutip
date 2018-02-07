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
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, spmv)

def _f1(t,args):
    return np.sin(t*args['w1'])

def _f2(t,args):
    return np.cos(t*args['w2'])

def _f3(t,args):
    return np.exp(1j*t*args['w3'])

def _random_td_Qobj(shape=(1,1), ops=[0,0,0], cte=True, tlist=None):
    """Create a list to make a td_Qobj with up to 3 coefficients"""
    if tlist is None:
        tlist = np.linspace(0,1,300)

    Qobj_list = []
    if cte:
        Qobj_list.append(Qobj(np.random.random(shape) + \
                              1j*np.random.random(shape)))

    coeff = [[_f1, "sin(w1*t)", np.sin(tlist)],
             [_f2, "cos(w2*t)", np.cos(tlist)],
             [_f3, "exp(w3*t*1j)", np.exp(tlist*1j)]]
    for i,form in enumerate(ops):
        if form:
            Qobj_list.append([Qobj(np.random.random(shape)),coeff[i][form-1]])
    return Qobj_list


def test_td_Qobj_cte_call():
    "td_Qobj call: cte"
    N = 5
    data = np.random.random((N, N))
    q1 = Qobj(data)
    td_data = td_Qobj(q1)
    # Check the constant flag
    assert_equal(td_data.const, True)
    # Check that the call return the Qobj
    assert_equal(td_data(np.random.random()) == q1, True)
    # Check that the call for the data return the data
    assert_equal((td_data(np.random.random(),data=True) - data == 0).all(), True)

    # Test another creation format
    data2 = np.random.random((N, N))
    q2 = Qobj(data2)
    td_data2 = td_Qobj([q2])
    # Check the constant flag
    assert_equal(td_data2.const, True)
    # Check that the call return the Qobj
    assert_equal(td_data2(np.random.random()) == q2, True)
    # Check that the call for the data return the data
    assert_equal((td_data2(np.random.random(), data=True) - data2 == 0).all(), True)


def test_td_Qobj_func_call():
    "td_Qobj call: func format"
    N = 5
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    data3 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)
    args={"w1":1, "w2":2}
    td_data = td_Qobj([q1, [q2,_f1], [q3,_f2]], args=args)
    # Check the constant flag
    assert_equal(td_data.const, False)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*args['w1']) * q2 + np.cos(t*args['w2']) * q3
    # Check that the call return the Qobj
    assert_equal(td_data(t) == q_at_t, True)
    data_at_t = data1 + np.sin(t*args['w1']) * data2 + np.cos(t*args['w2']) * data3
    # Check that the call for the data return the data
    assert_allclose(td_data(t, data=True).todense(), data_at_t)

    # test another init format
    td_data2 = td_Qobj([q1,_f1], args=args)
    # Check the constant flag
    assert_equal(td_data.const, False)
    t = np.random.random()
    q_at_t = q1 * np.sin(t*args['w1'])
    # Check that the call return the Qobj
    assert_equal(td_data2(t) == q_at_t, True)


def test_td_Qobj_str_call():
    "td_Qobj call: str format"
    N = 5
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    data3 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)
    args={"w1":1, "w2":2}
    td_data = td_Qobj([q1,[q2,'sin(t*w1)'],[q3,'cos(t*w2)']], args=args)
    # Check the constant flag
    assert_equal(td_data.const, False)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*args['w1']) * q2 + np.cos(t*args['w2']) * q3
    # Check that the call return the Qobj
    assert_equal(td_data(t) == q_at_t, True)
    data_at_t = data1 + np.sin(t*args['w1']) * data2 + np.cos(t*args['w2']) * data3
    # Check that the call for the data return the data
    assert_allclose(td_data(t,data=True).todense(), data_at_t)

    # test another init format
    td_data2 = td_Qobj([q1,'sin(t*w1)'], args=args)
    t = np.random.random()
    q_at_t = q1 * np.sin(t*args['w1'])
    # Check that the call return the Qobj
    assert_equal(td_data2(t) == q_at_t, True)


def test_td_Qobj_array_call():
    "td_Qobj call: array format"
    N = 5
    tlist = np.linspace(0,1,300)
    tlistlog = np.logspace(-3,0,600)
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    data3 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)
    td_data = td_Qobj([q1,[q2,np.sin(tlist)],[q3,np.cos(2*tlist)]], tlist=tlist)
    # Check the constant flag
    assert_equal(td_data.const, False)
    t = np.random.random() *.99 + 0.01
    q_at_t = q1 + np.sin(t) * q2 + np.cos(t*2) * q3
    # Check that the call return the Qobj
    assert_allclose(td_data(t,data=1).todense() , q_at_t.data.todense())
    data_at_t = data1 + np.sin(t) * data2 + np.cos(t*2) * data3
    # Check that the call for the data return the data
    assert_allclose(td_data(t,data=True).todense(), data_at_t)

    #test for non-linear tlist
    td_data = td_Qobj([q1,[q2,np.sin(tlistlog)],[q3,np.cos(2*tlistlog)]], tlist=tlistlog)
    t = np.random.random()
    data_at_t = data1 + np.sin(t) * data2 + np.cos(t*2) * data3
    # Check that the call for the data return the data
    assert_allclose(np.array(td_data(t,data=True).todense()), data_at_t, rtol=3e-7)


def test_td_Qobj_mixed_call():
    "td_Qobj call: mixed format"
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
    args={"w1":1, "w2":2}
    td_data = td_Qobj([q1, [q2,_f1], [q3,"cos(w2*t)"],
                      [q4,np.exp(3*tlist)]], tlist=tlist, args=args)
    t = np.random.random()
    data_at_t = data1 + np.sin(t) * data2 + np.cos(t*2) * data3 +\
                np.exp(t*3) * data4
    # Check that the call for the data return the data
    assert_allclose(td_data(t,data=True).todense(), data_at_t)


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


def test_td_Qobj_math_neg():
    "td_Qobj negation"
    tlist = np.linspace(0,1,300)
    td_obj = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_neg = -td_obj
    t = np.random.random()
    # check the negation
    assert_equal(td_obj(t) == -1*td_obj_neg(t), True)


def test_td_Qobj_math_add():
    "td_Qobj addition"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_2 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    q1 = Qobj(np.random.random((5,5)))
    td_obj_sum = td_obj_1 + td_obj_2
    t = np.random.random()
    # check the addition td_Qobj + td_Qobj
    assert_equal(td_obj_sum(t) == td_obj_1(t) + td_obj_2(t), True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 += td_obj_2
    # check the inplace addition td_Qobj += td_Qobj
    assert_equal(td_obj_sum2(t) == td_obj_1(t) + td_obj_2(t), True)
    td_obj_sum = td_obj_1 + q1
    t = np.random.random()
    # check the addition td_Qobj + Qobj
    assert_equal(td_obj_sum(t) == td_obj_1(t) + q1, True)
    td_obj_sum = q1 + td_obj_1
    t = np.random.random()
    # check the addition Qobj + td_Qobj
    assert_equal(td_obj_sum(t) == td_obj_1(t) + q1, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 += q1
    # check the inplace addition td_Qobj += Qobj
    assert_equal(td_obj_sum2(t) == td_obj_1(t) + q1, True)
    scalar = np.random.random()
    td_obj_sum = scalar + td_obj_1
    t = np.random.random()
    # check the addition td_Qobj + scalar
    assert_equal(td_obj_sum(t) == td_obj_1(t) + scalar, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 += scalar
    # check the inplace addition td_Qobj += scalar
    assert_equal(td_obj_sum2(t) == td_obj_1(t) + scalar, True)


def test_td_Qobj_math_sub():
    "td_Qobj subtraction"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_2 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    q1 = Qobj(np.random.random((5,5)))
    td_obj_sum = td_obj_1 - td_obj_2
    t = np.random.random()
    # check the subtraction -
    assert_equal(td_obj_sum(t) == td_obj_1(t) - td_obj_2(t), True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 -= td_obj_2
    # check the inplace subtraction -=
    assert_equal(td_obj_sum2(t) == td_obj_1(t) - td_obj_2(t), True)
    td_obj_sum = q1 - td_obj_1
    t = np.random.random()
    # check the addition td_Qobj - Qobj
    assert_equal(td_obj_sum(t) == q1 - td_obj_1(t), True)
    td_obj_sum = td_obj_1 - q1
    t = np.random.random()
    # check the addition td_Qobj - Qobj
    assert_equal(td_obj_sum(t) == td_obj_1(t) - q1, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 -= q1
    # check the inplace addition td_Qobj -= Qobj
    assert_equal(td_obj_sum2(t) == td_obj_1(t) - q1, True)
    scalar = np.random.random()
    td_obj_sum = scalar - td_obj_1
    t = np.random.random()
    # check the addition td_Qobj - scalar
    assert_equal(td_obj_sum(t) == -td_obj_1(t) + scalar, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 -= scalar
    # check the inplace addition td_Qobj -= scalar
    assert_equal(td_obj_sum2(t) == td_obj_1(t) - scalar, True)


def test_td_Qobj_math_mult():
    "td_Qobj multiplication"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    q1 = Qobj(np.random.random((5,5)))
    td_obj_sum = q1 * td_obj_1
    t = np.random.random()
    # check the addition td_Qobj * Qobj
    assert_equal(td_obj_sum(t) == q1 * td_obj_1(t), True)
    td_obj_sum = td_obj_1 *q1
    t = np.random.random()
    # check the addition td_Qobj * Qobj
    assert_equal(td_obj_sum(t) == td_obj_1(t) * q1, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 *= q1
    # check the inplace addition td_Qobj *= Qobj
    assert_equal(td_obj_sum2(t) == td_obj_1(t) * q1, True)
    scalar = np.random.random()
    td_obj_sum = scalar * td_obj_1
    t = np.random.random()
    # check the addition td_Qobj * scalar
    assert_equal(td_obj_sum(t) == td_obj_1(t) * scalar, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 *= scalar
    # check the inplace addition td_Qobj *= scalar
    assert_equal(td_obj_sum2(t) == td_obj_1(t) * scalar, True)


def test_td_Qobj_math_div():
    "td_Qobj division"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    scalar = np.random.random()
    td_obj_sum =  td_obj_1 / scalar
    t = np.random.random()
    # check the addition td_Qobj / scalar
    assert_equal(td_obj_sum(t) == td_obj_1(t) / scalar, True)
    td_obj_sum2 = td_obj_1.copy()
    td_obj_sum2 /= scalar
    # check the inplace addition td_Qobj /= scalar
    assert_equal(td_obj_sum2(t) == td_obj_1(t) / scalar, True)


def test_td_Qobj_trans():
    "td_Qobj transpose"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_trans =  td_obj_1.trans()
    t = np.random.random()
    # check the transpose
    assert_equal(td_obj_trans(t) == td_obj_1(t).trans(), True)


def test_td_Qobj_dag():
    "td_Qobj dag"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_dag =  td_obj_1.dag()
    t = np.random.random()
    # check the dag
    assert_equal(td_obj_dag(t) == td_obj_1(t).dag(), True)


def test_td_Qobj_conj():
    "td_Qobj conjugate"
    tlist = np.linspace(0,1,300)
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args={"w1":1, "w2":2}, tlist=tlist)
    td_obj_conj =  td_obj_1.conj()
    t = np.random.random()
    # check the conjugate
    assert_equal(td_obj_conj(t) == td_obj_1(t).conj(), True)


def test_td_Qobj_norm():
    "td_Qobj norm: a.dag * a"
    tlist = np.linspace(0,1,300)
    args={"w3":1}
    td_obj_1 = td_Qobj(_random_td_Qobj((5,5), [0,0,1], tlist=tlist, cte=0),
                       args=args, tlist=tlist)
    td_obj_2 = td_Qobj(_random_td_Qobj((5,5), [0,0,2], tlist=tlist, cte=0),
                       args=args, tlist=tlist)
    td_obj_3 = td_Qobj(_random_td_Qobj((5,5), [0,0,3], tlist=tlist, cte=0),
                       args=args, tlist=tlist)
    t = np.random.random()
    td_obj_norm =  td_obj_1.norm()
    # check the a.dag * dag, func
    assert_equal(td_obj_norm(t) == td_obj_1(t).dag() * td_obj_1(t), True)
    td_obj_norm =  td_obj_2.norm()
    # check the a.dag * dag, str
    assert_equal(td_obj_norm(t) == td_obj_2(t).dag() * td_obj_2(t), True)
    td_obj_norm =  td_obj_3.norm()
    # check the a.dag * dag, array
    assert_allclose(td_obj_norm(t,data=True).todense(),
                    (td_obj_3(t).dag() * td_obj_3(t)).data.todense())


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
    assert_equal(td_obj_2(t) == td_obj_1(t), True)


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
    assert_equal(td_obj_str_2.fast, True)
    assert_equal(td_obj_str_2(t) == td_list[0][0] * np.sin(t)**2, True)

    td_list = _random_td_Qobj((5,5), [3,0,0], tlist=tlist, cte=False)
    td_obj_array = td_Qobj(td_list, tlist=tlist)
    td_obj_array_2 = td_obj_array.apply_decorator(square_f, inplace_np=True)
    #Check that it can only be compiled to cython
    assert_equal(td_obj_array_2.fast, True)
    assert_allclose(np.array(td_obj_array_2(t, data=True).todense()),
                    np.array((td_list[0][0] * np.sin(t)**2).data.todense()),
                    rtol=3e-7)


def test_td_Qobj_compile():
    "td_Qobj compile"
    tlist = np.linspace(0, 1, 300)
    tlistlog = np.logspace(-3, 0, 600)
    args={"w1":1, "w2":2}
    td_obj_c = td_Qobj(_random_td_Qobj((5,5), [0,0,0]))
    td_obj_f = td_Qobj(_random_td_Qobj((5,5), [1,0,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_s = td_Qobj(_random_td_Qobj((5,5), [2,0,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_a = td_Qobj(_random_td_Qobj((5,5), [3,0,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_ca = td_Qobj(_random_td_Qobj((5,5), [0,0,3], tlist=tlistlog),
                        args=args, tlist=tlistlog)
    td_obj_sa = td_Qobj(_random_td_Qobj((5,5), [2,3,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_m = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)
    t = np.random.random() *.99 + 0.01
    td_obj_cc = td_obj_c.copy()
    td_obj_cc.compile()
    # check if compiled for const format
    assert_equal(td_obj_cc.compiled != 0, True)
    # check compiled for const format call
    assert_equal(td_obj_cc(t) == td_obj_c(t), True)
    td_obj_fc = td_obj_f.copy()
    td_obj_fc.compile()
    # check if compiled for func format
    assert_equal(td_obj_fc.compiled != 0, True)
    # check compiled for func format call
    assert_equal(td_obj_fc(t) == td_obj_f(t), True)
    td_obj_sc = td_obj_s.copy()
    td_obj_sc.compile()
    # check if compiled for str format
    assert_equal(td_obj_sc.compiled != 0, True)
    # check compiled for str format call
    assert_equal(td_obj_sc(t) == td_obj_s(t), True)
    td_obj_ac = td_obj_a.copy()
    td_obj_ac.compile()
    # check if compiled for array format
    assert_equal(td_obj_ac.compiled != 0, True)
    # check compiled for array format call
    assert_allclose(np.array(td_obj_ac(t,data=True).todense()), np.array(td_obj_a(t,data=True).todense()), rtol=3e-07, atol=0)
    td_obj_cac = td_obj_ca.copy()
    td_obj_cac.compile()
    # check if compiled for array format
    assert_equal(td_obj_cac.compiled != 0, True)
    # check compiled for array format call
    assert_allclose(np.array(td_obj_cac(t,data=True).todense()), np.array(td_obj_ca(t,data=True).todense()), rtol=3e-07, atol=0)
    td_obj_sac = td_obj_sa.copy()
    td_obj_sac.compile()
    # check if compiled for mixed array str format
    assert_equal(td_obj_sac.compiled != 0, True)
    # check compiled for mixed array str format call
    assert_allclose(np.array(td_obj_sac(t,data=True).todense()), np.array(td_obj_sa(t,data=True).todense()), rtol=3e-07, atol=0)
    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    # check if compiled for mixed format
    assert_equal(td_obj_mc.compiled != 0, True)
    # check compiled for mixed format call
    assert_equal(td_obj_mc(t) == td_obj_m(t), True)
    assert_allclose(np.array(td_obj_ac(t,data=True).todense()), np.array(td_obj_a(t,data=True).todense()), rtol=3e-07, atol=0)


def test_td_Qobj_rhs():
    "td_Qobj rhs"
    tlist = np.linspace(0,1,300)
    args={"w1":1, "w2":2}
    td_obj_c = td_Qobj(_random_td_Qobj((5,5), [0,0,0]))
    td_obj_f = td_Qobj(_random_td_Qobj((5,5), [1,0,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_sa = td_Qobj(_random_td_Qobj((5,5), [2,3,0], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_m = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)
    t = np.random.random()
    vec = np.arange(5)*.5+.5j

    td_obj_cc = td_obj_c.copy()
    td_obj_cc.compile()
    v1 = td_obj_c.rhs(t,vec)
    v2 = td_obj_cc.rhs(t,vec)
    v3 = spmv(td_obj_c(t, data=True), vec)
    # check not compiled rhs const
    assert_allclose(v1, v3)
    # check compiled rhs
    assert_allclose(v3, v2)

    td_obj_fc = td_obj_f.copy()
    td_obj_fc.compile()
    v1 = td_obj_f.rhs(t,vec)
    v2 = td_obj_fc.rhs(t,vec)
    v3 = spmv(td_obj_f(t, data=True), vec)
    # check not compiled rhs func
    assert_allclose(v1, v3)
    # check compiled rhs func
    assert_allclose(v3, v2)

    td_obj_sac = td_obj_sa.copy()
    td_obj_sac.compile()
    v1 = td_obj_sa.rhs(t,vec)
    v2 = td_obj_sac.rhs(t,vec)
    v3 = spmv(td_obj_sa(t, data=True), vec)
    # check not compiled rhs array str
    assert_allclose(v1, v3)
    # check compiled rhs array str
    assert_allclose(v3, v2)

    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    v1 = td_obj_m.rhs(t,vec)
    v2 = td_obj_mc.rhs(t,vec)
    v3 = spmv(td_obj_m(t, data=True), vec)
    # check not compiled rhs mixed
    assert_allclose(v1, v3)
    # check compiled rhs mixed
    assert_allclose(v3, v2)


def test_td_Qobj_expect_psi():
    "td_Qobj expect psi"
    tlist = np.linspace(0,1,300)
    args={"w1":1, "w2":2, "w3":3}
    td_obj_c = td_Qobj(_random_td_Qobj((5,5), [0,0,0]))
    td_obj_f = td_Qobj(_random_td_Qobj((5,5), [0,0,1], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_sa = td_Qobj(_random_td_Qobj((5,5), [0,3,2], tlist=tlist),
                       args=args, tlist=tlist)
    td_obj_m = td_Qobj(_random_td_Qobj((5,5), [1,2,3], tlist=tlist),
                       args=args, tlist=tlist)
    t = np.random.random()
    vec = np.arange(5)*.5+.5j

    td_obj_cc = td_obj_c.copy()
    td_obj_cc.compile()
    v1 = td_obj_c.expect(t, vec, 0)
    v2 = td_obj_cc.expect(t, vec, 0)
    v3 = cy_expect_psi(td_obj_c(t, data=True), vec, 0)
    # check not compiled rhs const
    assert_allclose(v1, v3)
    # check compiled rhs
    assert_allclose(v3, v2)

    td_obj_fc = td_obj_f.copy()
    td_obj_fc.compile()
    v1 = td_obj_f.expect(t, vec, 1)
    v2 = td_obj_fc.expect(t, vec, 1)
    v3 = cy_expect_psi(td_obj_f(t, data=True), vec, 1)
    # check not compiled rhs func
    assert_allclose(v1, v3)
    # check compiled rhs func
    assert_allclose(v3, v2)

    td_obj_sac = td_obj_sa.copy()
    td_obj_sac.compile()
    v1 = td_obj_sa.expect(t, vec, 0)
    v2 = td_obj_sac.expect(t, vec, 0)
    v3 = cy_expect_psi(td_obj_sa(t, data=True), vec, 0)
    # check not compiled rhs array str
    assert_allclose(v1, v3)
    # check compiled rhs array str
    assert_allclose(v3, v2)

    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    v1 = td_obj_m.expect(t, vec, 1)
    v2 = td_obj_mc.expect(t ,vec, 1)
    v3 = cy_expect_psi(td_obj_m(t, data=True), vec, 1)
    # check not compiled rhs mixed
    assert_allclose(v1, v3)
    # check compiled rhs mixed
    assert_allclose(v3, v2)


def test_td_Qobj_expect():
    "td_Qobj expect rho"
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
    assert_allclose(v1, v3)
    # check compiled rhs
    assert_allclose(v3, v2)

    td_obj_mc = td_obj_m.copy()
    td_obj_mc.compile()
    v1 = td_obj_m.expect(t, rho, 1)
    v2 = td_obj_mc.expect(t, rho, 1)
    v3 = cy_expect_rho_vec(td_obj_m(t, data=True), rho, 1)
    # check not compiled rhs func
    assert_allclose(v1, v3)
    # check compiled rhs func
    assert_allclose(v3, v2)


def test_td_Qobj_func_args():
    "td_Qobj args: func arguments changes"
    N = 5
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    data3 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)
    args={"w1":1, "w2":2}
    td_data = td_Qobj([q1,[q2,_f1],[q3,_f2]], args=args)

    t = np.random.random()
    q_at_t = q1 + np.sin(t*args['w1']) * q2 + np.cos(t*10) * q3
    # Check that the call with custom args
    assert_equal(td_data.with_args(t,{"w2":10}) == q_at_t, True)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*args['w1']) * q2 + np.cos(t*args['w2']) * q3
    # Check that the with_args call did not change the original args
    assert_equal(td_data(t) == q_at_t, True)

    new_args={"w1":10}
    td_data.arguments(new_args)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*10) * q2 + np.cos(t*args['w2']) * q3
    # Check the arguments change
    assert_equal(td_data(t) == q_at_t, True)

    td_data.compile()
    new_args={"w1":5}
    td_data.arguments(new_args)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*5) * q2 + np.cos(t*args['w2']) * q3
    # Check the arguments change after compile
    assert_equal(td_data(t) == q_at_t, True)


def test_td_Qobj_str_args():
    "td_Qobj args: str arguments changes"
    N = 5
    data1 = np.random.random((N, N))
    data2 = np.random.random((N, N))
    data3 = np.random.random((N, N))
    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)
    args={"w1":1, "w2":2}
    td_data = td_Qobj([q1,[q2,"sin(w1*t)"],[q3,"cos(w2*t)"]], args=args)

    t = np.random.random()
    q_at_t = q1 + np.sin(t*args['w1']) * q2 + np.cos(t*10) * q3
    # Check that the call with custom args
    assert_equal(td_data.with_args(t,{"w2":10}) == q_at_t, True)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*args['w1']) * q2 + np.cos(t*args['w2']) * q3
    # Check that the with_args call did not change the original args
    assert_equal(td_data(t) == q_at_t, True)

    new_args={"w1":10}
    td_data.arguments(new_args)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*10) * q2 + np.cos(t*args['w2']) * q3
    # Check the arguments change
    assert_equal(td_data(t) == q_at_t, True)

    td_data.compile()
    new_args={"w1":5}
    td_data.arguments(new_args)
    t = np.random.random()
    q_at_t = q1 + np.sin(t*5) * q2 + np.cos(t*args['w2']) * q3
    # Check the arguments change after compile
    assert_equal(td_data(t) == q_at_t, True)


def test_td_Qobj_mixed_args():
    "td_Qobj args: mixed arguments changes"
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
    args={"w1":1, "w2":2}
    td_data = td_Qobj([q1, [q2,_f1], [q3,"cos(w2*t)"],
                      [q4,np.exp(3*tlist)]], tlist=tlist, args=args)

    t = np.random.random()
    data_at_t = data1 + np.sin(t*args['w1']) * data2 +\
             np.cos(t*10) * data3 + np.exp(3*t) * data4
    # Check that the call with custom args
    assert_allclose(np.array(td_data.with_args(t,{"w2":10},data=True).todense()), data_at_t)

    t = np.random.random()
    data_at_t = data1 + np.sin(t*10) * data2 +\
             np.cos(t*args['w2']) * data3 + np.exp(3*t) * data4
    # Check that the call with custom args
    assert_allclose(np.array(td_data.with_args(t,{"w1":10},data=True).todense()), data_at_t)

    t = np.random.random()
    data_at_t = data1 + np.sin(t*args['w1']) * data2 +\
             np.cos(t*args['w2']) * data3 + np.exp(3*t) * data4
    # Check that the with_args call did not change the original args
    assert_allclose(np.array(td_data(t,data=True).todense()), data_at_t)

    new_args={"w1":10, "w2":7}
    td_data.arguments(new_args)
    t = np.random.random()
    data_at_t = data1 + np.sin(t*10) * data2 + np.cos(t*7) * data3 + np.exp(3*t) * data4
    # Check the arguments change
    assert_allclose(np.array(td_data(t,data=True).todense()), data_at_t)

    td_data.compile()
    new_args={"w1":3, "w2":5}
    td_data.arguments(new_args)
    t = np.random.random()
    data_at_t = data1 + np.sin(t*3) * data2 + np.cos(t*5) * data3 + np.exp(3*t) * data4
    # Check the arguments change after compile
    assert_allclose(np.array(td_data(t,data=True).todense()), data_at_t)


def test_td_Qobj_with_state():
    "td_Qobj args: with_state"
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


def test_td_Qobj_pickle_cy_td_Qobj():
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

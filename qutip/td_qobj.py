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
"""Time-dependent Quantum Object (Qobj) class.
"""
__all__ = ['td_Qobj']#, 'td_liouvillian', 'td_lindblad_dissipator']

from qutip.qobj import Qobj
from scipy.interpolate import CubicSpline
from functools import partial, wraps
from types import FunctionType, BuiltinFunctionType
import numpy as np
from numbers import Number
from qutip.td_qobj_codegen import _compile_str_single, make_united_f_ptr
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, spmv)
from qutip.cy.td_qobj_cy import cy_cte_qobj, cy_td_qobj
import pickle

class td_Qobj:
    """A class for representing time-dependent quantum objects,
    such as quantum operators and states.

    The td_Qobj class is a representation of time-dependent Qutip quantum
    objects (Qobj). This class implements math operations :
        +,- : td_Qobj, Qobj
        * : Qobj, C-number
        / : C-number
    and some common linear operator/state operations. The td_Qobj
    are constructed from a nested list of Qobj with their time-dependent
    coefficients. The time-dependent coefficients are either a funciton, a
    string or a numpy array.

    For function format, the function signature must be f(t, args).
    *Examples*
        def f1_t(t,args):
            return np.exp(-1j*t*args["w1"])

        def f2_t(t,args):
            return np.cos(t*args["w2"])

        H = td_Qobj([H0, [H1, f1_t], [H2, f2_t]], args={"w1":1.,"w2":2.})

    For string based, the string must be a compilable python code resulting in
    a scalar. The following symbols are defined:
        sin cos tan asin acos atan pi
        sinh cosh tanh asinh acosh atanh
        exp log log10 erf sqrt
        real imag conj abs norm arg proj
    numpy is also imported as np.
    *Examples*
        H = td_Qobj([H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
                    args={"w1":1.,"w2":2.})

    For numpy array format, the array must be an 1d of dtype float64 or complex.
    A list of times (float64) at which the coeffients must be given (tlist).
    The coeffients array must have the same len as the tlist.
    The time of the tlist do not need to be equidistant, but must be sorted.
    *Examples*
        tlist = np.logspace(-5,0,100)
        H = td_Qobj([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)

    Mixing the formats is possible, but not recommended.

    Parameters
    ----------
    td_Qobj(Q_object=[], args={}, tlist=None, raw_str=False)
    Q_object : array_like
        Data for vector/matrix representation of the quantum object.
    args : dictionary that contain the arguments for
    tlist : array_like
        List of times at which the numpy-array coefficients are applied. Times
        must be equidistant and start from 0.
    raw_str : delay the compilation of str based coefficient until the "compile"
        method is called.


    Attributes
    ----------
    cte : Qobj
        Constant part of the td_Qobj
    ops : list
        List of Qobj and the coefficients.
        [(Qobj, coefficient as a function, original coefficient,
            type, local arguments ), ... ]
        type :
            1: function
            2: string
            3: np.array
    args : map
        arguments of the coefficients
    tlist : array_like
        List of times at which the numpy-array coefficients are applied.

    compiled : int
        Has the cython version of the td_Qobj been created
    compiled_Qobj : cy_qobj (cy_cte_qobj or cy_td_qobj)
        Cython version of the td_Qobj
    dummy_cte : bool
        is self.cte a dummy Qobj
    const : bool
        Indicates if quantum object is Constant
    raw_str : bool
        compile the str coefficient only at the cration of the cy_qobj
    fast : bool
        Use a cython function for the coefficient of the cy_qobj? (str/array)
    N_obj : int
        number of Qobj in the td_Qobj : len(ops) + (1 if not dummy_cte)


    Methods
    -------
    copy() :
        Create copy of Qobj
    arguments(new_args):
        Update the args of the object

    Math:
        +/- td_Qobj, Qobj, scalar:
            Addition is possible between td_Qobj and with Qobj or scalar
        -:
            Negation operator
        * Qobj, scalar:
            Product is possible with Qobj or scalar
        / scalar:
            It is possible to divide by scalar only
    conj()
        Return the conjugate of quantum object.
    dag()
        Return the adjoint (dagger) of quantum object.
    trans()
        Return the transpose of quantum object.
    norm()
        Return self.dag() * self.
        Only possible if N_obj == 1
    permute(order)
        Returns composite qobj with indices reordered.
    ptrace(sel)
        Returns quantum object for selected dimensions after performing
        partial trace.
    apply(f, *args, **kw_args)
        Apply the function f to every Qobj. f(Qobj) -> Qobj
        Return a modified td_Qobj and let the original one untouched
    apply_decorator(decorator, *args, str_mod=None,
                    inplace_np=False, **kw_args):
        Apply the decorator to each function of the ops.
        The *args and **kw_args are passed to the decorator.
        new_coeff_function = decorator(coeff_function, *args, **kw_args)
        str_mod : list of 2 elements
            replace the string : str_mod[0] + original_string + str_mod[1]
            *exemple: str_mod = ["exp(",")"]
        inplace_np:
            Change the numpy array instead of applying the decorator to the
            function reading the array. Some decorators create incorrect array.
            Transformations f'(t) = f(g(t)) create a missmatch between the array
            and the associated time list.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    compress():
        Merge ops which are based on the same quantum object and coeff type.

    compile():
        Create the associated cython object for faster usage.
    __call__(t):
        Return the Qobj at time t.
        *Faster after compilation
    with_args(t, new_args):
        Return the Qobj at time t with the new_args instead of the original
        arguments. Do not change the args of the object.
    with_state(t, psi, args={}):
        Allow to use function coefficients that use states:
            "def coeff(t,psi,args):" instead of "def coeff(t,args):"
        Return the Qobj at time t, with the new_args if defined.
        *Mixing both definition types of coeff will make the td_Qobj crach on
            call
    rhs(t, psi):
        Apply the quantum object (if operator, no check) to psi.
        More generaly, return the product of the object at t with psi.
        *Faster after compilation
    expect(t, psi, herm=False):
        Calculates the expectation value for the quantum object (if operator,
            no check) and state psi.
        Return only the real part if herm.
        *Faster after compilation
    get_compiled_call, get_rhs_func, get_expect_func:
        Compile and return the corresponding function of the cython object.
        Was useful before the python function was set to call the cython version
            after compilation.
    to_list():
        Return the time-dependent quantum object as a list
    """

    def __init__(self, Q_object=[], args={}, tlist=None, raw_str=False):
        if isinstance(Q_object, td_Qobj):
            self._inplace_copy(Q_object)
            if args:
                self.arguments(args)
            return

        self.const = False
        self.dummy_cte = False
        self.args = args
        self.cte = None
        self.tlist = tlist
        self.fast = True
        self.compiled = False
        self.compiled_Qobj = None
        self.compiled_ptr = None
        self.raw_str = raw_str

        if isinstance(Q_object, list) and len(Q_object) == 2:
            if isinstance(Q_object[0], Qobj) and not isinstance(Q_object[1],
                                                                (Qobj, list)):
                # The format is [Qobj, f/str]
                Q_object = [Q_object]

        op_type = self._td_format_check_single(Q_object, tlist)
        self.ops = []

        if isinstance(op_type, int):
            if op_type == 0:
                self.cte = Q_object
                self.const = True
            elif op_type == 1:
                raise Exception("The Qobj must not already be a function")
            elif op_type == -1:
                pass

        else:
            compile_list = []
            compile_count = 0
            for type_, op in zip(op_type, Q_object):
                if type_ == 0:
                    if self.cte is None:
                        self.cte = op
                    else:
                        self.cte += op
                elif type_ == 1:
                    self.ops.append([op[0], op[1], op[1], 1, args])
                    self.fast = False
                elif type_ == 2:
                    local_args = {}
                    for i in args:
                        if i in op[1]:
                            local_args[i] = args[i]
                    self.ops.append([op[0], _dummy, op[1], 2, local_args])
                    compile_list.append((op[1], local_args, compile_count))
                    compile_count += 1
                elif type_ == 3:
                    l = len(self.ops)
                    N = len(self.tlist)
                    dt = self.tlist[-1] / (N - 1)
                    self.ops.append([op[0], CubicSpline(tlist, op[1]),
                                     op[1].copy(), 3, None])

                else:
                    raise Exception("Should never be here")

            if compile_count and not self.raw_str:
                str_funcs = _compile_str_single(compile_list)
                count = 0
                for op in self.ops:
                    if op[3] == 2:
                        op[1] = str_funcs[count]
                        count += 1

            if not self.cte:
                self.cte = self.ops[0][0]
                for op in self.ops[1:]:
                    self.cte += op[0]
                self.cte *= 0.
                self.dummy_cte = True

            if not self.ops:
                self.const = True
        self.N_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)

    def _td_format_check_single(self, Q_object, tlist=None):
        op_type = []

        if isinstance(Q_object, Qobj):
            op_type = 0
        elif isinstance(Q_object, (FunctionType,
                                   BuiltinFunctionType, partial)):
            op_type = 1
        elif isinstance(Q_object, list):
            if (len(Q_object) == 0):
                op_type = -1
            for op_k in Q_object:
                if isinstance(op_k, Qobj):
                    op_type.append(0)
                elif isinstance(op_k, list):
                    if not isinstance(op_k[0], Qobj):
                        raise TypeError("Incorrect Q_object specification")
                    elif len(op_k) == 2:
                        if isinstance(op_k[1], (FunctionType,
                                                BuiltinFunctionType, partial)):
                            op_type.append(1)
                        elif isinstance(op_k[1], str):
                            op_type.append(2)
                        elif isinstance(op_k[1], np.ndarray):
                            if not isinstance(tlist, np.ndarray) or not \
                                        len(op_k[1]) == len(tlist):
                                raise TypeError("Time list do not match")
                            op_type.append(3)
                        elif isinstance(op_k[1], Cubic_Spline):
                            raise NotImplementedError(
                                    "Cubic_Spline not supported")
                        #    h_obj.append(k)
                        else:
                            raise TypeError("Incorrect Q_object specification")
                    else:
                        raise TypeError("Incorrect Q_object specification")
        else:
            raise TypeError("Incorrect Q_object specification")
        return op_type

    def __call__(self, t, data=False):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if self.compiled:
            return self.compiled_Qobj.call(t, data)
        if data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                if part[3] == 1:  # func: f(t,args)
                    op_t += part[0].data * part[1](t, part[4])
                elif part[3] == 2:  # str: f(t,w=2)
                    op_t += part[0].data * part[1](t, **part[4])
                elif part[3] == 3:  # numpy: _interpolate(t,arr,N,dt)
                    op_t += part[0].data * part[1](np.array([t]))[0]
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                if part[3] == 1:  # func: f(t,args)
                    op_t += part[0] * part[1](t, part[4])
                elif part[3] == 2:  # str: f(t,w=2)
                    op_t += part[0] * part[1](t, **part[4])
                elif part[3] == 3:  # numpy: _interpolate(t,arr,N,dt)
                    op_t += part[0] * part[1](np.array([t]))[0]
        return op_t

    def with_args(self, t, args, data=False):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        coeff = np.zeros(len(self.ops), dtype=complex)
        new_args = self.args.copy()
        new_args.update(args)
        if self.compiled:
            for i, part in enumerate(self.ops):
                if part[3] == 1:  # func: f(t,args)
                    coeff[i] = part[1](t, new_args)
                elif part[3] == 2:  # str: f(t,w=2)
                    part_args = part[4].copy()
                    for pa in part_args:
                        if pa in args:
                            part_args[pa] = new_args[pa]
                    coeff[i] = part[1](t, **part_args)
                elif part[3] == 3:  # numpy: _interpolate(t,arr,N,dt)
                    coeff[i] = part[1](np.array([t]))[0]
            op_t = self.compiled_Qobj.call_with_coeff(t, coeff, data=data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                if part[3] == 1:
                    op_t += part[0].data * part[1](t, new_args)
                elif part[3] == 2:
                    part_args = part[4].copy()
                    for pa in part_args:
                        if pa in args:
                            part_args[pa] = new_args[pa]
                    op_t += part[0].data * part[1](t, **part_args)
                elif part[3] == 3:
                    op_t += part[0].data * part[1](np.array([t]))[0]
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                if part[3] == 1:
                    op_t += part[0] * part[1](t, new_args)
                elif part[3] == 2:
                    part_args = part[4].copy()
                    for pa in part_args:
                        if pa in args:
                            part_args[pa] = new_args[pa]
                    op_t += part[0] * part[1](t, **part_args)
                elif part[3] == 3:
                    op_t += part[0] * part[1](np.array([t]))[0]
        return op_t

    def with_state(self, t, psi, args={}, data=False):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        if self.compiled == 3:
            coeff = np.zeros(len(self.ops), dtype=complex)
            if args:
                new_args = self.args.copy()
                new_args.update(args)
                for i, part in enumerate(self.ops):
                    coeff[i] = part[1](t, psi, new_args)
            else:
                for i, part in enumerate(self.ops):
                    coeff[i] = part[1](t, psi, part[4])
            op_t = self.compiled_Qobj.call_with_coeff(t, coeff, data=data)
        elif self.compiled:
            coeff = np.zeros(len(self.ops), dtype=complex)
            if args:
                new_args = self.args.copy()
                new_args.update(args)
                for i, part in enumerate(self.ops):
                    if part[3] == 1:  # func: f(t, psi, args)
                        coeff[i] = part[1](t, psi, new_args)
                    elif part[3] == 2:  # str: f(t,w=2)
                        part_args = part[4].copy()
                        for pa in part_args:
                            if pa in new_args:
                                part_args[pa] = new_args[pa]
                        coeff[i] = part[1](t, **part_args)
                    elif part[3] == 3:  # numpy: scipy's cubic spline
                        coeff[i] = part[1](np.array([t]))[0]
            else:
                for i, part in enumerate(self.ops):
                    if part[3] == 1:
                        coeff[i] = part[1](t, psi, part[4])
                    elif part[3] == 2:
                        coeff[i] = part[1](t, **part[4])
                    elif part[3] == 3:
                        coeff[i] = part[1](np.array([t]))[0]
            op_t = self.compiled_Qobj.call_with_coeff(t, coeff, data=data)
        elif args:
            new_args = self.args.copy()
            new_args.update(args)
            if data:
                op_t = self.cte.data.copy()
                for part in self.ops:
                    if part[3] == 1:
                        op_t += part[0].data * part[1](t, psi, new_args)
                    elif part[3] == 2:
                        part_args = part[4].copy()
                        for pa in part_args:
                            if pa in args:
                                part_args[pa] = args[pa]
                        op_t += part[0].data * part[1](t, **part_args)
                    elif part[3] == 3:
                        op_t += part[0].data * part[1](np.array([t]))[0]
            else:
                op_t = self.cte.copy()
                for part in self.ops:
                    if part[3] == 1:
                        op_t += part[0] * part[1](t, psi, new_args)
                    elif part[3] == 2:
                        part_args = part[4].copy()
                        for pa in part_args:
                            if pa in args:
                                part_args[pa] = args[pa]
                        op_t += part[0] * part[1](t, **part_args)
                    elif part[3] == 3:
                        op_t += part[0] * part[1](np.array([t]))[0]
        else:
            if data:
                op_t = self.cte.data.copy()
                for part in self.ops:
                    if part[3] == 1:
                        op_t += part[0].data * part[1](t, psi, part[4])
                    elif part[3] == 2:
                        op_t += part[0].data * part[1](t, **part[4])
                    elif part[3] == 3:
                        op_t += part[0].data * part[1](np.array([t]))[0]
            else:
                op_t = self.cte.copy()
                for part in self.ops:
                    if part[3] == 1:
                        op_t += part[0] * part[1](t, psi, part[4])
                    elif part[3] == 2:
                        op_t += part[0] * part[1](t, **part[4])
                    elif part[3] == 3:
                        op_t += part[0] * part[1](np.array([t]))[0]
        return op_t

    def copy(self):
        new = td_Qobj(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.tlist = self.tlist
        new.dummy_cte = self.dummy_cte
        new.N_obj = self.N_obj
        new.fast = self.fast
        new.raw_str = self.raw_str
        new.compiled = False
        new.compiled_Qobj = None
        new.compiled_ptr = None
        new.coeff_get = None

        for l, op in enumerate(self.ops):
            new.ops.append([None, None, None, None, None])
            new.ops[l][0] = op[0].copy()
            new.ops[l][3] = op[3]
            new.ops[l][1] = op[1]
            if new.ops[l][3] in [1, 2]:
                new.ops[l][2] = op[2]
                new.ops[l][4] = op[4].copy()
            elif new.ops[l][3] == 3:
                new.ops[l][2] = op[2].copy()
                new.ops[l][4] = None

        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.tlist = other.tlist
        self.dummy_cte = other.dummy_cte
        self.N_obj = other.N_obj
        self.fast = other.fast
        self.raw_str = other.raw_str
        self.compiled = False
        self.compiled_Qobj = None
        self.compiled_ptr = None
        self.coeff_get = None
        self.ops = []

        for l, op in enumerate(other.ops):
            self.ops.append([None, None, None, None, None])
            self.ops[l][0] = op[0].copy()
            self.ops[l][3] = op[3]
            self.ops[l][1] = op[1]
            if self.ops[l][3] in [1, 2]:
                self.ops[l][2] = op[2]
                self.ops[l][4] = op[4].copy()
            elif self.ops[l][3] == 3:
                self.ops[l][2] = op[2].copy()
                self.ops[l][4] = None

    def arguments(self, args):
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(args)
        if self.compiled == 2:
            self.coeff_get(True).set_args(self.args)
        for op in self.ops:
            if op[3] == 1:
                op[4] = self.args
            elif op[3] == 2:
                local_args = {}
                for i in self.args:
                    if i in op[2]:
                        local_args[i] = self.args[i]
                op[4] = local_args

    def to_list(self):
        list_Qobj = []
        if not self.dummy_cte:
            list_Qobj.append(self.cte)
        for op in self.ops:
            if op[3] == 1:
                list_Qobj.append([op[0], op[1]])
            else:
                list_Qobj.append([op[0], op[2]])
        return list_Qobj

    # Math function
    def __add__(self, other):
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        res = self.copy()
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, td_Qobj):
            self.cte += other.cte
            l = len(self.ops)
            for op in other.ops:
                self.ops.append([None, None, None, None, None])
                self.ops[l][0] = op[0].copy()
                self.ops[l][3] = op[3]
                self.ops[l][1] = op[1]
                if self.ops[l][3] in [1, 2]:
                    self.ops[l][2] = op[2]
                    self.ops[l][4] = op[4].copy()
                elif self.ops[l][3] == 3:
                    self.ops[l][2] = op[2].copy()
                    self.ops[l][4] = None
                l += 1
            self.args.update(**other.args)
            self.const = self.const and other.const
            self.dummy_cte = self.dummy_cte and other.dummy_cte
            self.fast = self.fast and other.fast
            self.raw_str = self.raw_str and other.raw_str
            self.compiled = False
            self.compiled_Qobj = None
            self.compiled_ptr = None
            self.coeff_get = None

            if self.tlist is None:
                self.tlist = other.tlist
            else:
                if other.tlist is None:
                    pass
                elif len(other.tlist) != len(self.tlist) or \
                        other.tlist[-1] != self.tlist[-1]:
                    raise Exception("tlist are not compatible")
        else:
            self.cte += other
            self.dummy_cte = False

        self.N_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)
        return self

    def __sub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):
        res = -self.copy()
        res += other
        return res

    def __isub__(self, other):
        self += (-other)
        return self

    def __mul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):
        res = self.copy()
        if isinstance(other, Qobj):
            res.cte = other * res.cte
            for op in res.ops:
                op[0] = other * op[0]
            return res
        else:
            res *= other
            return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, Number):
            self.cte *= other
            for op in self.ops:
                op[0] *= other
        elif isinstance(other, td_Qobj):
            if other.const:
                self.cte *= other.cte
                for op in self.ops:
                    op[0] *= other.cte
            elif self.const:
                cte = self.cte.copy()
                self = other.copy()
                self.cte *= cte
                for op in self.ops:
                    op[0] *= cte
            else:
                raise Exception("When multiplying td_qobj, one " +
                                "of them must be constant")
        else:
            raise TypeError("td_qobj can only be multiplied" +
                            " with Qobj or numbers")
        return self

    def __div__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            res = self.copy()
            res *= other**(-1)
            return res
        else:
            raise TypeError('Incompatible object for division')

    def __idiv__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            self *= other**(-1)
        else:
            raise TypeError('Incompatible object for division')
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        res = self.copy()
        res.cte = (-res.cte)
        for op in res.ops:
            op[0] = -op[0]
        return res

    # Transformations
    def trans(self):
        res = self.copy()
        res.cte = res.cte.trans()
        for op in res.ops:
            op[0] = op[0].trans()
        return res

    def conj(self):
        res = self.copy()
        res.cte = res.cte.conj()
        for op in res.ops:
            op[0] = op[0].conj()
        res._f_conj()
        return res

    def dag(self):
        res = self.copy()
        res.cte = res.cte.dag()
        for op in res.ops:
            op[0] = op[0].dag()
        res._f_conj()
        return res

    # When name fixed, put in stochastic/ mcsolve
    def norm(self):
        """return a.dag * a """
        if not self.N_obj == 1:
            raise NotImplementedError("Only possible with one composing Qobj")
        res = self.copy()
        res.cte = res.cte.dag() * res.cte
        for op in res.ops:
            op[0] = op[0].dag() * op[0]
        res._f_norm2()
        return res

    # Unitary function of Qobj
    def tidyup(self, atol=1e-12):
        self.cte = self.cte.tidyup(atol)
        for op in self.ops:
            op[0] = op[0].tidyup(atol)
        return self

    def compress(self):
        sets = []
        for i, op1 in enumerate(self.ops):
            already_matched = False
            for _set in sets:
                already_matched = already_matched or i in _set
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1[0] == op2[0] and op1[3] == op2[3]:
                        this_set.append(j+i+1)
                sets.append(this_set)
        if len(self.ops) != len(sets):
            # found 2 td part with the same Qobj
            self.compiled = False
            self.compiled_Qobj = None
            self.compiled_ptr = None
            self.coeff_get = None
            new_ops = []
            for _set in sets:
                if len(_set) == 1:
                    new_ops.append(self.ops[_set[0]])
                elif self.ops[_set[0]][3] == 1:
                    new_fs = [self.ops[_set[0]][1]]
                    new_args = self.ops[_set[0]][4].copy()
                    for i in _set[1:]:
                        new_fs += [self.ops[i][1]]
                        new_args.update(self.ops[i][4])
                    new_op = [self.ops[_set[0]][0], None, new_fs, 1, new_args]
                    def _new_f(t, *args):
                        return sum((f(t, *args) for f in new_fs))
                    new_op[1] = _new_f
                    new_ops.append(new_op)

                elif self.ops[_set[0]][3] == 2:
                    new_op = self.ops[_set[0]]
                    new_str = self.ops[_set[0]][2]
                    for i in _set[1:]:
                        if self.ops[i][4]:
                            new_op[4].update(self.ops[i][4])
                        new_str = "(" + new_str + ") + (" + \
                                  self.ops[i][2] + ")"
                    if self.raw_str:
                        new_op[1] = _dummy
                    else:
                        new_op[1] = \
                            _compile_str_single([[new_str, new_op[4], 0]])[0]
                    new_op[2] = new_str
                    new_ops.append(new_op)
                elif self.ops[_set[0]][3] == 3:
                    new_op = self.ops[_set[0]]
                    new_array = (self.ops[_set[0]][2]).copy()
                    for i in _set[1:]:
                        new_array += self.ops[i][2]
                    new_op[2] = new_array
                    new_op[1] = CubicSpline(self.tlist, new_array)
                    new_op[4] = None
                    new_ops.append(new_op)
            self.ops = new_ops

    def permute(self, order):
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op[0] = op[0].permute(order)
        return res

    def ptrace(self, sel):
        res = self.copy()
        res.cte = res.cte.ptrace(sel)
        for op in res.ops:
            op[0] = op[0].ptrace(sel)
        return res

    # function to apply custom transformations
    def apply(self, function, *args, **kw_args):
        self.compiled = False
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op[0] = function(op[0], *args, **kw_args)
        return res

    def apply_decorator(self, function, *args, **kw_args):
        if "str_mod" in kw_args:
            str_mod = kw_args["str_mod"]
            del kw_args["str_mod"]
        else:
            str_mod = None
        if "inplace_np" in kw_args:
            inplace_np  = kw_args["inplace_np"]
            del kw_args["inplace_np"]
        else:
            inplace_np = None
        self.compiled = False
        res = self.copy()
        raw_str = self.raw_str
        for op in res.ops:
            if op[3] == 1:
                op[1] = function(op[1], *args, **kw_args)
                op[2] = function(op[1], *args, **kw_args)
            if op[3] == 2:
                if str_mod is None:
                    if self.raw_str:
                        op[1] = _compile_str_single([[op[2], op[4], 0]])[0]
                        raw_str = False
                    op[1] = function(op[1], *args, **kw_args)
                    res.fast = False
                else:
                    if not self.raw_str:
                        op[1] = function(op[1], *args, **kw_args)
                    op[2] = str_mod[0] + op[2] + str_mod[1]
            elif op[3] == 3:
                if inplace_np:
                    # keep the original function, change the array
                    def f(a):
                        return a
                    ff = function(f, *args, **kw_args)
                    for i, v in enumerate(op[2]):
                        op[2][i] = ff(v)
                    op[1] = CubicSpline(self.tlist, op[2])
                else:
                    op[1] = function(op[1], *args, **kw_args)
                    res.fast = False
        self.raw_str = raw_str
        return res

    def _f_norm2(self):
        for op in self.ops:
            if op[3] == 1:
                op[1] = _norm2(op[1])
                op[2] = op[1]
            elif op[3] == 2:
                op[2] = "norm(" + op[2] + ")"
                if not self.raw_str:
                    op[1] = _compile_str_single([[op[2], op[4], 0]])[0]
            elif op[3] == 3:
                op[2] = np.abs(op[2])**2
                op[1] = CubicSpline(self.tlist, op[2])
        return self

    def _f_conj(self):
        for op in self.ops:
            if op[3] == 1:
                op[1] = _conj(op[1])
                op[2] = op[1]
            elif op[3] == 2:
                op[2] = "conj(" + op[2] + ")"
                if not self.raw_str:
                    op[1] = _compile_str_single([[op[2], op[4], 0]])[0]
            elif op[3] == 3:
                op[2] = np.conj(op[2])
                op[1] = CubicSpline(self.tlist, op[2])
        return self

    def get_compiled_call(self):
        if not self.compiled:
            self.compile()
        return self.compiled_Qobj.call

    def get_rhs_func(self):
        if not self.compiled:
            self.compile()
        return self.compiled_Qobj.rhs

    def get_expect_func(self):
        if not self.compiled:
            self.compile()
        return self.compiled_Qobj.expect

    def expect(self, t, vec, herm=0):
        if not isinstance(t, (int, float)):
            raise TypeError("The time need to be a real scalar")
        if isinstance(vec, Qobj):
            if self.cte.dims[1] != vec.dims[0]:
                raise Exception("Dimensions do not fit")
            vec = vec.full().ravel()
        elif not isinstance(vec, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if vec.ndim != 1:
            raise Exception("The vector must be 1d")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")
        if not isinstance(herm, (int, bool)):
            herm = bool(herm)
        if self.compiled:
            return self.compiled_Qobj.expect(t, vec, herm)
        if self.cte.issuper:
            return cy_expect_rho_vec(self.__call__(t, data=True), vec, herm)
        else:
            return cy_expect_psi(self.__call__(t, data=True), vec, herm)

    def rhs(self, t, vec):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(vec, Qobj):
            if self.cte.dims[1] != vec.dims[0]:
                raise Exception("Dimensions do not fit")
            vec = vec.full().ravel()
        elif not isinstance(vec, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if vec.ndim != 1:
            raise Exception("The vector must be 1d")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")
        if self.compiled:
            return self.compiled_Qobj.rhs(t, vec)
        return spmv(self.__call__(t, data=True), vec)

    def compile(self, code=False):
        self.tidyup()
        if self.const:
            self.compiled_Qobj = cy_cte_qobj()
            self.compiled_Qobj.set_data(self.cte)
            self.compiled = 1
        elif self.fast:
            self.compiled_Qobj = cy_td_qobj()
            self.compiled_Qobj.set_data(self.cte, self.ops)
            if code:
                self.coeff_get, Code = make_united_f_ptr(self.ops,
                                                    self.args, self.tlist,
                                                    True)
            else:
                self.coeff_get = make_united_f_ptr(self.ops, self.args,
                                                          self.tlist, False)
                Code = None
            self.compiled_Qobj.set_factor(ptr=self.coeff_get())
            self.compiled = 2
            return Code
        else:
            self.compiled_Qobj = cy_td_qobj()
            self.compiled_Qobj.set_data(self.cte, self.ops)
            self.coeff_get, self.compiled = self._make_united_f_call()
            self.compiled_Qobj.set_factor(func=self.coeff_get)

    def _make_united_f_call(self):
        types = [0, 0, 0]
        for part in self.ops:
            types[part[3]-1] += 1
        if sum(types) == 0:
            if len(self.ops) == 0:
                raise Exception("No td operator but constant flag missing")
            else:
                raise Exception("Type of td_operator not supported")
        elif types[1] == 0 and types[2] == 0:
            # Only functions
            self.funclist = []
            for part in self.ops:
                self.funclist.append(part[1])
            united_f_call = _united_f_caller(self.funclist, self.args)
            """def united_f_call(t):
                out = []
                for func in self.funclist:
                    out.append(func(t, self.args))
                return out"""
            all_function = 3
        else:
            # Must be mixed, would be fast otherwise
            united_f_call  = self._get_coeff
            """def united_f_call(t):
                out = []
                for part in self.ops:
                    if part[3] == 1:  # func: f(t,args)
                        out.append(part[1](t, part[4]))
                    elif part[3] == 2:  # str: f(t,w=2)
                        if self.raw_str:
                            # Must compile the str here
                            part[1] = \
                                _compile_str_single([[part[2], part[4], 0]])[0]
                        out.append(part[1](t, **part[4]))
                    elif part[3] == 3:  # numpy: _interpolate(t,arr,N,dt)
                        out.append(part[1](np.array([t]))[0])
                return out"""
            all_function = 4
        return united_f_call, all_function

    def _get_coeff(self, t):
        out = []
        for part in self.ops:
            if part[3] == 1:  # func: f(t,args)
                out.append(part[1](t, part[4]))
            elif part[3] == 2:  # str: f(t,w=2)
                if self.raw_str:
                    # Must compile the str here
                    part[1] = \
                        _compile_str_single([[part[2], part[4], 0]])[0]
                out.append(part[1](t, **part[4]))
            elif part[3] == 3:  # numpy: _interpolate(t,arr,N,dt)
                out.append(part[1](np.array([t]))[0])
        return out

    def __getstate__(self):
        _dict_ = {key: self.__dict__[key]
                  for key in self.__dict__ if key is not "compiled_Qobj"}
        if self.compiled:
            return (_dict_, self.compiled_Qobj.__getstate__())
            #return (_dict_, pickle.dumps(self.compiled_Qobj))
        else:
            return (_dict_,)

    def __setstate__(self, state):
        self.__dict__ = state[0]
        if not self.compiled:
            self.compiled_Qobj = None
        elif self.compiled == 1:
            self.compiled_Qobj = cy_cte_qobj.__new__(cy_cte_qobj)
            self.compiled_Qobj.__setstate__(state[1])
            #self.compiled_Qobj = pickle.loads(state[1])
        elif self.compiled in (2,3,4):
            self.compiled_Qobj = cy_td_qobj.__new__(cy_td_qobj)
            self.compiled_Qobj.__setstate__(state[1])
            #self.compiled_Qobj = pickle.loads(state[1])


#Function defined inside another function cannot be pickle,
#Using class instead
class _united_f_caller:
    def __init__(self, funclist, args):
        self.funclist = funclist
        self.args = args

    def __call__ (self,t):
        out = []
        for func in self.funclist:
            out.append(func(t, self.args))
        return out


class _compress_f_caller:
    def __init__(self, funclist):
        self.funclist = funclist

    def __call__ (self, t, *args):
        return sum((f(t, *args) for f in self.funclist))


def _dummy(t, *args, **kwargs):
    return 0.


def _norm2(f):
    @wraps(f)
    def ff(a, *args, **kwargs):
        return np.abs(f(a, *args, **kwargs))**2
    return ff


def _conj(f):
    @wraps(f)
    def ff(a, *args, **kwargs):
        return np.conj(f(a, *args, **kwargs))
    return ff


"""def td_liouvillian(H, c_ops=[], chi=None, args={}, tlist=None, raw_str=False):
    ""Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators. Accept time dependant
    operator and return a td_qobj

    Parameters
    ----------
    H : qobj, [qobj], td_Qobj
        System Hamiltonian.

    c_ops : array_like of qobj or td_Qobj
        A ``list`` or ``array`` of collapse operators.

    args, tlist, raw_str:
        Arguments to pass to the td_qobj

    Returns
    -------
    L : td_qobj
        Liouvillian superoperator.

    ""
    L = None

    if chi and len(chi) != len(c_ops):
        raise ValueError('chi must be a list with same length as c_ops')

    if H is not None:
        if not isinstance(H, td_Qobj):
            L = td_Qobj(H, args=args, tlist=tlist, raw_str=raw_str)
        else:
            L = H
        L = L.apply(liouvillian, chi=chi)

    if isinstance(c_ops, list) and len(c_ops) > 0:
        def liouvillian_c(c_ops, chi):
            return liouvillian(None, c_ops=[c_ops], chi=chi)
        for c in c_ops:
            if not isinstance(c, td_Qobj):
                cL = td_Qobj(c, args=args, tlist=tlist, raw_str=raw_str)
            else:
                cL = c
            if not cL.N_obj == 1:
                raise Exception("Each c_ops must be composed of  " +
                                "only one Qobj to be used " +
                                "with in a time-dependent liouvillian")

            if L is None:
                L = cL.apply(liouvillian_c, chi=chi)._f_norm2()

            else:
                L += cL.apply(liouvillian_c, chi=chi)._f_norm2()

    return L


def td_lindblad_dissipator(a, args={}, tlist=None, raw_str=False):
    ""
    Lindblad dissipator (generalized) for a single collapse operator.
    For the

    .. math::

        \\mathcal{D}[a,b]\\rho = a \\rho b^\\dagger -
        \\frac{1}{2}a^\\dagger b\\rho - \\frac{1}{2}\\rho a^\\dagger b

    Parameters
    ----------
    a : qobj, [qobj], td_Qobj
        Left part of collapse operator.

    args, tlist, raw_str:
        Arguments to pass to the td_qobj

    Returns
    -------
    D : td_qobj
        Lindblad dissipator superoperator.
    ""
    if not isinstance(a, td_Qobj):
        b = td_Qobj(a, args=args, tlist=tlist, raw_str=raw_str)
    else:
        b = a
    if not b.N_obj == 1:
        raise Exception("Each sc_ops must be composed of only one Qobj to " +
                        "be used with in a time-dependent lindblad_dissipator")

    D = b.apply(lindblad_dissipator)._f_norm2()
    return D"""

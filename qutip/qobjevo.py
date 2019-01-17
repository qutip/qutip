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
__all__ = ['QobjEvo']

from qutip.qobj import Qobj
import qutip.settings as qset
from qutip.interpolate import Cubic_Spline
from scipy.interpolate import CubicSpline
from functools import partial, wraps
from types import FunctionType, BuiltinFunctionType
import numpy as np
from numbers import Number
from qutip.qobjevo_codegen import _compile_str_single, _compiled_coeffs
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, spmv)
from qutip.cy.cqobjevo import (CQobjCte, CQobjCteDense, CQobjEvoTd,
                                 CQobjEvoTdMatched, CQobjEvoTdDense)
from qutip.cy.cqobjevo_factor import (InterCoeffT, InterCoeffCte,
                                      InterpolateCoeff)
import pickle
import sys
import scipy

if qset.has_openmp:
    from qutip.cy.openmp.cqobjevo_omp import (CQobjCteOmp, CQobjEvoTdOmp,
                                              CQobjEvoTdMatchedOmp)

safePickle = False
if sys.platform == 'win32':
    safePickle = True


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf+0j


str_env = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "pi": np.pi,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "erf": scipy.special.erf,
    "sqrt": np.sqrt,
    "real": np.real,
    "imag": np.imag,
    "conj": np.conj,
    "abs": np.abs,
    "norm": lambda x: np.abs(x)**2,
    "arg": np.angle,
    "proj": proj,
    "np": np}


class _StrWrapper:
    def __init__(self, code):
        self.code = "_out = " + code

    def __call__(self, t, args={}):
        env = {"t": t}
        env.update(args)
        exec(self.code, str_env, env)
        return env["_out"]


class _CubicSplineWrapper:
    # Using scipy's CubicSpline since Qutip's one
    # only accept linearly distributed tlist
    def __init__(self, tlist, coeff):
        self.coeff = coeff
        self.tlist = tlist
        self.func = CubicSpline(self.tlist, self.coeff)

    def __call__(self, t, args={}):
        return self.func([t])[0]


class QobjEvo:
    """A class for representing time-dependent quantum objects,
    such as quantum operators and states.

    The QobjEvo class is a representation of time-dependent Qutip quantum
    objects (Qobj). This class implements math operations :
        +,- : QobjEvo, Qobj
        * : Qobj, C-number
        / : C-number
    and some common linear operator/state operations. The QobjEvo
    are constructed from a nested list of Qobj with their time-dependent
    coefficients. The time-dependent coefficients are either a funciton, a
    string or a numpy array.

    For function format, the function signature must be f(t, args).
    *Examples*
        def f1_t(t,args):
            return np.exp(-1j*t*args["w1"])

        def f2_t(t,args):
            return np.cos(t*args["w2"])

        H = QobjEvo([H0, [H1, f1_t], [H2, f2_t]], args={"w1":1.,"w2":2.})

    For string based, the string must be a compilable python code resulting in
    a scalar. The following symbols are defined:
        sin cos tan asin acos atan pi
        sinh cosh tanh asinh acosh atanh
        exp log log10 erf sqrt
        real imag conj abs norm arg proj
    numpy is also imported as np.
    *Examples*
        H = QobjEvo([H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
                    args={"w1":1.,"w2":2.})

    For numpy array format, the array must be an 1d of dtype float or complex.
    A list of times (float64) at which the coeffients must be given (tlist).
    The coeffients array must have the same len as the tlist.
    The time of the tlist do not need to be equidistant, but must be sorted.
    *Examples*
        tlist = np.logspace(-5,0,100)
        H = QobjEvo([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)

    Mixing the formats is possible, but not recommended.
    Mixing tlist will cause problem.

    Parameters
    ----------
    QobjEvo(Q_object=[], args={}, tlist=None)
    Q_object : array_like
        Data for vector/matrix representation of the quantum object.
    args : dictionary that contain the arguments for
    tlist : array_like
        List of times at which the numpy-array coefficients are applied. Times
        must be equidistant and start from 0.

    Attributes
    ----------
    cte : Qobj
        Constant part of the QobjEvo
    ops : list
        List of Qobj and the coefficients.
        [(Qobj, coefficient as a function, original coefficient,
            type, local arguments ), ... ]
        type :
            1: function
            2: string
            3: np.array
            4: Cubic_Spline
    args : map
        arguments of the coefficients
    tlist : array_like
        List of times at which the numpy-array coefficients are applied.

    compiled : int
        Has the cython version of the QobjEvo been created
    compiled_Qobj : cy_qobj (CQobjCte or CQobjEvoTd)
        Cython version of the QobjEvo
    dummy_cte : bool
        is self.cte a dummy Qobj
    const : bool
        Indicates if quantum object is Constant
    type : int
        todo
    N_obj : int
        number of Qobj in the QobjEvo : len(ops) + (1 if not dummy_cte)


    Methods
    -------
    copy() :
        Create copy of Qobj
    arguments(new_args):
        Update the args of the object

    Math:
        +/- QobjEvo, Qobj, scalar:
            Addition is possible between QobjEvo and with Qobj or scalar
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
        Return a modified QobjEvo and let the original one untouched
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
            Transformations f'(t) = f(g(t)) create a missmatch between the
            array and the associated time list.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    compress():
        Merge ops which are based on the same quantum object and coeff type.

    compile(code=False, matched=False, dense=False, omp=0):
        Create the associated cython object for faster usage.
        code: return the code generated for compilation of the strings.
        matched: the compiled object use sparse matrix with matching indices.
                    (experimental, no real advantage)
        dense: the compiled object use dense matrix.
        omp: (int) number of thread: the compiled object use spmvpy_openmp.

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
        *Mixing both definition types of coeff will make the QobjEvo fail on
            call
    mul_mat(t, mat):
        Product of this at t time with the dense matrix mat.
        *Faster after compilation
    mul_vec(t, psi):
        Apply the quantum object (if operator, no check) to psi.
        More generaly, return the product of the object at t with psi.
        *Faster after compilation
    expect(t, psi, herm=False):
        Calculates the expectation value for the quantum object (if operator,
            no check) and state psi.
        Return only the real part if herm.
        *Faster after compilation
    to_list():
        Return the time-dependent quantum object as a list
    """

    def __init__(self, Q_object=[], args={}, tlist=None):
        if isinstance(Q_object, QobjEvo):
            self._inplace_copy(Q_object)
            if args:
                self.arguments(args)
            return

        self.const = False
        self.dummy_cte = False
        self.args = args
        self.cte = None
        self.tlist = tlist
        self.compiled = False
        self.compiled_Qobj = None
        self.compiled_ptr = None
        self.coeff_get = None
        self.type = -1
        self.omp = 0
        self.coeff_files = []

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
                self.type = 0
            elif op_type == 1:
                raise Exception("The Qobj must not already be a function")
            elif op_type == -1:
                pass

        else:
            op_type_count = [0, 0, 0, 0]
            for type_, op in zip(op_type, Q_object):
                if type_ == 0:
                    if self.cte is None:
                        self.cte = op
                    else:
                        self.cte += op
                elif type_ == 1:
                    op_type_count[0] += 1
                    self.ops.append([op[0], op[1], op[1], 1])
                elif type_ == 2:
                    op_type_count[1] += 1
                    self.ops.append([op[0], _StrWrapper(op[1]), op[1], 2])
                elif type_ == 3:
                    op_type_count[2] += 1
                    self.ops.append([op[0], _CubicSplineWrapper(tlist, op[1]),
                                     op[1].copy(), 3])
                elif type_ == 4:
                    op_type_count[3] += 1
                    self.ops.append([op[0], op[1], op[1], 4])
                else:
                    raise Exception("Should never be here")

            nops = sum(op_type_count)
            if op_type_count[0] == nops:
                self.type = 1
            elif op_type_count[1] == nops:
                self.type = 2
            elif op_type_count[2] == nops:
                self.type = 3
            elif op_type_count[3] == nops:
                self.type = 4
            elif op_type_count[0]:
                self.type = 5
            else:
                self.type = 6

            try:
                if not self.cte:
                    self.cte = self.ops[0][0]
                    for op in self.ops[1:]:
                        self.cte += op[0]
                    self.cte *= 0.
                    self.dummy_cte = True
                else:
                    cte_copy = self.cte.copy()
                    for op in self.ops:
                        cte_copy += op[0]
            except Exception as err:
                raise Exception("Qobj not compatible:\n" + err)

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
                        if isinstance(op_k[1], Cubic_Spline):
                            op_type.append(4)
                        elif callable(op_k[1]):
                            op_type.append(1)
                        elif isinstance(op_k[1], str):
                            op_type.append(2)
                        elif isinstance(op_k[1], np.ndarray):
                            if not isinstance(tlist, np.ndarray) or not \
                                        len(op_k[1]) == len(tlist):
                                raise TypeError("Time list does not match")
                            op_type.append(3)
                        else:
                            raise TypeError("Incorrect Q_object specification")
                    else:
                        raise TypeError("Incorrect Q_object specification")
        else:
            raise TypeError("Incorrect Q_object specification")
        return op_type

    def __del__(self):
        for filename in self.coeff_files:
            try:
                os.remove(filename+".pyx")
            except:
                pass

    def __call__(self, t, data=False):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if self.const:
            if data:
                op_t = self.cte.data.copy()
            else:
                op_t = self.cte.copy()
        elif self.compiled and self.compiled//10 != 2:
            op_t = self.compiled_Qobj.call(t, data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part[0].data * part[1](t, self.args)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part[0] * part[1](t, self.args)
        return op_t

    def with_args(self, t, args, data=False):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        new_args = self.args.copy()
        new_args.update(args)
        if self.const:
            if data:
                op_t = self.cte.data.copy()
            else:
                op_t = self.cte.copy()
        elif self.compiled and self.compiled//10 != 2:
            coeff = np.zeros(len(self.ops), dtype=complex)
            for i, part in enumerate(self.ops):
                coeff[i] = part[1](t, new_args)
            op_t = self.compiled_Qobj.call_with_coeff(coeff, data=data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part[0].data * part[1](t, new_args)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part[0] * part[1](t, new_args)
        return op_t

    def with_state(self, t, psi, args={}, data=False):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        if args:
            new_args = self.args.copy()
            new_args.update(args)
        else:
            new_args = self.args
        if self.type not in [1, 5]:
            # no pure function than can accept state
            if args:
                op_t = self.with_args(t, args, data)
            else:
                op_t = self.__call__(t, data)
        elif self.type == 1:
            if self.compiled:
                coeff = np.zeros(len(self.ops), dtype=complex)
                for i, part in enumerate(self.ops):
                    coeff[i] = part[1](t, psi, new_args)
                op_t = self.compiled_Qobj.call_with_coeff(coeff, data=data)
            else:
                if data:
                    op_t = self.cte.data.copy()
                    for part in self.ops:
                        op_t += part[0].data * part[1](t, psi, new_args)
                else:
                    op_t = self.cte.copy()
                    for part in self.ops:
                        op_t += part[0] * part[1](t, psi, new_args)
        else:
            coeff = np.zeros(len(self.ops), dtype=complex)
            for i, part in enumerate(self.ops):
                if part[3] == 1:  # func: f(t, psi, args)
                    coeff[i] = part[1](t, psi, new_args)
                else:
                    coeff[i] = part[1](t, new_args)
            if self.compiled and self.compiled//10 != 2:
                op_t = self.compiled_Qobj.call_with_coeff(coeff, data=data)
            else:
                if data:
                    op_t = self.cte.data.copy()
                    for c, part in zip(coeff, self.ops):
                        op_t += part[0].data * c
                else:
                    op_t = self.cte.copy()
                    for c, part in zip(coeff, self.ops):
                        op_t += part[0] * c

        return op_t

    def copy(self):
        new = QobjEvo(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.tlist = self.tlist
        new.dummy_cte = self.dummy_cte
        new.N_obj = self.N_obj
        new.type = self.type
        new.compiled = False
        new.compiled_Qobj = None
        new.compiled_ptr = None
        new.coeff_get = None

        for l, op in enumerate(self.ops):
            new.ops.append([None, None, None, None])
            new.ops[l][0] = op[0].copy()
            new.ops[l][1] = op[1]
            new.ops[l][3] = op[3]
            if new.ops[l][3] == 3:
                new.ops[l][2] = op[2].copy()
            else:
                new.ops[l][2] = op[2]

        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.tlist = other.tlist
        self.dummy_cte = other.dummy_cte
        self.N_obj = other.N_obj
        self.type = other.type
        self.compiled = False
        self.compiled_Qobj = None
        self.compiled_ptr = None
        self.coeff_get = None
        self.ops = []

        for l, op in enumerate(other.ops):
            self.ops.append([None, None, None, None])
            self.ops[l][0] = op[0].copy()
            self.ops[l][3] = op[3]
            self.ops[l][1] = op[1]
            if self.ops[l][3] == 3:
                self.ops[l][2] = op[2].copy()
            else:
                self.ops[l][2] = op[2]

    def arguments(self, args):
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(args)
        if self.compiled in [2, 12, 22, 32, 42, 3, 13, 23, 33, 43]:
            self.coeff_get.set_args(self.args)

    def to_list(self):
        list_Qobj = []
        if not self.dummy_cte:
            list_Qobj.append(self.cte)
        for op in self.ops:
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
        if isinstance(other, QobjEvo):
            self.cte += other.cte
            l = len(self.ops)
            for op in other.ops:
                self.ops.append([None, None, None, None])
                self.ops[l][0] = op[0].copy()
                self.ops[l][3] = op[3]
                self.ops[l][1] = op[1]
                if self.ops[l][3] == 3:
                    self.ops[l][2] = op[2].copy()
                else:
                    self.ops[l][2] = op[2]
                l += 1
            self.args.update(**other.args)
            self.const = self.const and other.const
            self.dummy_cte = self.dummy_cte and other.dummy_cte
            if self.type != other.type:
                if self.type in [1, 5] or other.type in [1, 5]:
                    self.type = 5
                else:
                    self.type = 6
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
        self._reset_type()
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
        elif isinstance(other, QobjEvo):
            if other.const:
                self.cte *= other.cte
                for op in self.ops:
                    op[0] *= other.cte
            elif self.const:
                cte = self.cte.copy()
                self = other.copy()
                self.cte = cte * self.cte
                for op in self.ops:
                    op[0] = cte*op[0]
            else:
                cte = self.cte.copy()
                self.cte *= other.cte
                new_terms = []
                old_ops = self.ops
                if not other.dummy_cte:
                    for op in old_ops:
                        new_terms.append(self._ops_mul_cte(op, other.cte, "R"))
                if not self.dummy_cte:
                    for op in other.ops:
                        new_terms.append(self._ops_mul_cte(op, cte, "L"))

                for op_left in old_ops:
                    for op_right in other.ops:
                        new_terms.append(self._ops_mul_(op_left,
                                                        op_right))
                self.ops = new_terms
                self.args.update(other.args)
                self.dummy_cte = self.dummy_cte and other.dummy_cte
                self.N_obj = (len(self.ops) if
                              self.dummy_cte else len(self.ops) + 1)
            self._reset_type()

        else:
            raise TypeError("td_qobj can only be multiplied" +
                            " with td_qobj, Qobj or numbers")
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

    def _ops_mul_(self, opL, opR):
        new_f = _Prod(opL[1], opR[1])
        new_op = [opL[0]*opR[0], new_f, None, 0]
        if opL[3] == opR[3] and opL[3] == 2:
            new_op[2] = "("+opL[2]+") * ("+opR[2]+")"
            new_op[3] = 2
        elif opL[3] == opR[3] and opL[3] == 3:
            new_op[2] = opL[2]*opR[2]
            new_op[3] = 3
        else:
            new_op[2] = new_f
            new_op[3] = 1
            if self.type not in [1, 5]:
                self.type = 5
        return new_op

    def _ops_mul_cte(self, op, cte, side):
        new_op = [None, op[1], op[2], op[3]]
        if side == "R":
            new_op[0] = op[0] * cte
        if side == "L":
            new_op[0] = cte * op[0]
        return new_op

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

    def norm(self):
        """return a.dag * a """
        if not self.N_obj == 1:
            res = self.dag()
            res *= self
        else:
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

    def _compress_make_set(self):
        sets = []
        for i, op1 in enumerate(self.ops):
            already_matched = False
            for _set in sets:
                already_matched = already_matched or i in _set
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1[0] == op2[0] and (op1[3] == op2[3] or
                            (op1[3] in [1, 4] and op2[3] in [1, 4])):
                        this_set.append(j+i+1)
                sets.append(this_set)

        fsets = []
        for i, op1 in enumerate(self.ops):
            already_matched = False
            for _set in fsets:
                already_matched = already_matched or i in _set
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1[3] in [1, 2, 4] and op2[3] == op1[3]:
                        if op1[2] is op2[2]:
                            this_set.append(j+i+1)
                    elif op1[3] in [3] and op2[3] == op1[3]:
                        if np.allclose(op1[2], op2[2]):
                            this_set.append(j+i+1)
                fsets.append(this_set)
        return sets, fsets

    def _compress_merge_Qobj(self, sets):
        new_ops = []
        for _set in sets:
            if len(_set) == 1:
                new_op = self.ops[_set[0]]

            elif self.ops[_set[0]][3] in [1, 4]:
                new_op = [self.ops[_set[0]][0], None, None, 1]
                fs = []
                for i in _set:
                    fs += [self.ops[i][1]]
                new_op[1] = _Add(fs)
                new_op[2] = new_op[1]

            elif self.ops[_set[0]][3] == 2:
                new_op = [self.ops[_set[0]][0], None, None, 2]
                new_str = "(" + self.ops[_set[0]][2] + ")"
                for i in _set[1:]:
                    new_str += " + (" + self.ops[i][2] + ")"
                new_op[1] = _StrWrapper(new_str)
                new_op[2] = new_str

            elif self.ops[_set[0]][3] == 3:
                new_op = [self.ops[_set[0]][0], None, None, 2]
                new_array = (self.ops[_set[0]][2]).copy()
                for i in _set[1:]:
                    new_array += self.ops[i][2]
                new_op[2] = new_array
                new_op[1] = _CubicSplineWrapper(self.tlist, new_array)

            new_ops.append(new_op)
        self.ops = new_ops

    def _compress_merge_func(self, fsets):
        new_ops = []
        for _set in fsets:
            if len(_set) == 1:
                new_op = self.ops[_set[0]]
            else:
                new_op = self.ops[_set[0]]
                new_op[0] = self.ops[_set[0]][0].copy()
                for i in _set[1:]:
                    new_op[0] += self.ops[i][0]
            new_ops.append(new_op)
        self.ops = new_ops

    def compress(self):
        self.tidyup()
        sets, fsets = self._compress_make_set()
        N_sets = len(sets)
        N_fsets = len(fsets)
        N_ops = len(self.ops)

        if N_sets < N_ops and N_fsets < N_ops:
            # Both could be better
            self.compiled = False
            self.compiled_Qobj = None
            self.coeff_get = None
            if N_sets < N_fsets:
                self._compress_merge_Qobj(sets)
            else:
                self._compress_merge_func(fsets)
            sets, fsets = self._compress_make_set()
            N_sets = len(sets)
            N_fsets = len(fsets)
            N_ops = len(self.ops)

        if N_sets < N_ops:
            self.compiled = False
            self.compiled_Qobj = None
            self.coeff_get = None
            self._compress_merge_Qobj(sets)
        elif N_fsets < N_ops:
            self.compiled = False
            self.compiled_Qobj = None
            self.coeff_get = None
            self._compress_merge_func(fsets)
        self._reset_type()

    def _reset_type(self):
        op_type_count = [0, 0, 0, 0]
        for op in self.ops:
            if op[3] == 1:
                op_type_count[0] += 1
            elif op[3] == 2:
                op_type_count[1] += 1
            elif op[3] == 3:
                op_type_count[2] += 1
            elif op[3] == 4:
                op_type_count[3] += 1

        nops = sum(op_type_count)
        if op_type_count[0] == nops:
            self.type = 1
        elif op_type_count[1] == nops:
            self.type = 2
        elif op_type_count[2] == nops:
            self.type = 3
        elif op_type_count[3] == nops:
            self.type = 4
        elif op_type_count[0]:
            self.type = 5
        else:
            self.type = 6

        self.N_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)

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
            inplace_np = kw_args["inplace_np"]
            del kw_args["inplace_np"]
        else:
            inplace_np = None
        res = self.copy()
        for op in res.ops:
            op[1] = function(op[1], *args, **kw_args)
            if op[3] == [1, 4]:
                op[2] = function(op[1], *args, **kw_args)
                op[3] = 1
            elif op[3] == 2:
                if str_mod is None:
                    op[2] = op[1]
                    op[3] = 1
                else:
                    op[2] = str_mod[0] + op[2] + str_mod[1]
            elif op[3] == 3:
                if inplace_np:
                    # keep the original function, change the array
                    def f(a):
                        return a
                    ff = function(f, *args, **kw_args)
                    for i, v in enumerate(op[2]):
                        op[2][i] = ff(v)
                    op[1] = _CubicSplineWrapper(self.tlist, op[2])
                else:
                    op[2] = function(op[1], *args, **kw_args)
                    op[3] = 1
        if self.type == 2 and str_mod is None:
            res.type = 5
        elif self.type == 3 and not inplace_np:
            res.type = 5
        elif self.type == 4:
            res.type = 5
        elif self.type == 6:
            for op in res.ops:
                if op[3] == 1:
                    res.type = 5
        return res

    def _f_norm2(self):
        for op in self.ops:
            if op[3] == 1:
                op[1] = _Norm2(op[1])
                op[2] = op[1]
            elif op[3] == 2:
                op[2] = "norm(" + op[2] + ")"
                op[1] = _StrWrapper(op[2])
            elif op[3] == 3:
                op[2] = np.abs(op[2])**2
                op[1] = _CubicSplineWrapper(self.tlist, op[2])
            elif op[3] == 4:
                op[1] = _Norm2(op[1])
                op[2] = op[1]
                op[3] = 1
                self.type = 5
        return self

    def _f_conj(self):
        for op in self.ops:
            if op[3] == 1:
                op[1] = _Conj(op[1])
                op[2] = op[1]
            elif op[3] == 2:
                op[2] = "conj(" + op[2] + ")"
                op[1] = _StrWrapper(op[2])
            elif op[3] == 3:
                op[2] = np.conj(op[2])
                op[1] = _CubicSplineWrapper(self.tlist, op[2])
            elif op[3] == 4:
                op[1] = _Conj(op[1])
                op[2] = op[1]
                op[3] = 1
                self.type = 5
        return self

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

    def mul_vec(self, t, vec):
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(vec, Qobj):
            if self.cte.dims[1] != vec.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = vec.dims
            vec = vec.full().ravel()
        elif not isinstance(vec, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if vec.ndim != 1:
            raise Exception("The vector must be 1d")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")
        if self.compiled:
            out = self.compiled_Qobj.mul_vec(t, vec)
        else:
            out = spmv(self.__call__(t, data=True), vec)
        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def mul_mat(self, t, mat):
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if not isinstance(mat, np.ndarray):
            raise TypeError("The vector must be an array")
        if mat.ndim != 2:
            raise Exception("The matrice must be 2d")
        if mat.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")
        if self.compiled:
            out = self.compiled_Qobj.mul_mat(t, mat)
        else:
            out = self.__call__(t, data=True) * mat
        return out

    def compile(self, code=False, matched=False, dense=False, omp=0):
        self.tidyup()
        Code = None
        if not qset.has_openmp:
            omp = 0
        if self.const:
            if dense:
                self.compiled_Qobj = CQobjCteDense()
                self.compiled = 21
            elif omp:
                self.compiled_Qobj = CQobjCteOmp()
                self.compiled = 31
                self.compiled_Qobj.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_Qobj = CQobjCte()
                self.compiled = 1
            self.compiled_Qobj.set_data(self.cte)
        else:
            if matched:
                if omp:
                    self.compiled_Qobj = CQobjEvoTdMatchedOmp()
                    self.compiled = 40
                    self.compiled_Qobj.set_threads(omp)
                    self.omp = omp
                else:
                    self.compiled_Qobj = CQobjEvoTdMatched()
                    self.compiled = 10
            elif dense:
                self.compiled_Qobj = CQobjEvoTdDense()
                self.compiled = 20
            elif omp:
                self.compiled_Qobj = CQobjEvoTdOmp()
                self.compiled = 30
                self.compiled_Qobj.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_Qobj = CQobjEvoTd()
                self.compiled = 0
            self.compiled_Qobj.set_data(self.cte, self.ops)
            if self.type in [1]:
                funclist = []
                for part in self.ops:
                    funclist.append(part[1])
                self.coeff_get = _UnitedFuncCaller(funclist, self.args)
                self.compiled += 2
                self.compiled_Qobj.set_factor(func=self.coeff_get)
            elif self.type in [5]:
                funclist = []
                for part in self.ops:
                    if isinstance(part[1], _StrWrapper):
                        part[1], file = _compile_str_single(part[2], self.args)
                        self.coeff_files.append(file)
                    funclist.append(part[1])
                self.coeff_get = _UnitedFuncCaller(funclist, self.args)
                self.compiled += 2
                self.compiled_Qobj.set_factor(func=self.coeff_get)
            elif self.type in [2, 6]:
                # All factor can be compiled
                self.coeff_get, Code, file = _compiled_coeffs(self.ops,
                                                              self.args,
                                                              self.tlist)
                self.coeff_files.append(file)
                self.compiled_Qobj.set_factor(obj=self.coeff_get)
                self.compiled += 3
            elif self.type == 3:
                if np.allclose(np.diff(self.tlist),
                               self.tlist[1] - self.tlist[0]):
                    self.coeff_get = InterCoeffCte(self.ops, None,
                                                     self.tlist)
                else:
                    self.coeff_get = InterCoeffT(self.ops, None, self.tlist)
                self.compiled += 3
                self.compiled_Qobj.set_factor(obj=self.coeff_get)
            elif self.type == 4:
                self.coeff_get = InterpolateCoeff(self.ops, None, None)
                self.compiled += 3
                self.compiled_Qobj.set_factor(obj=self.coeff_get)
            else:
                pass
            if code:
                return Code

    def _get_coeff(self, t):
        out = []
        for part in self.ops:
            out.append(part[1](t, self.args))
        return out

    def __getstate__(self):
        _dict_ = {key: self.__dict__[key]
                  for key in self.__dict__ if key is not "compiled_Qobj"}
        if self.compiled:
            return (_dict_, self.compiled_Qobj.__getstate__())
        else:
            return (_dict_,)

    def __setstate__(self, state):
        self.__dict__ = state[0]
        self.compiled_Qobj = None
        if self.compiled:
            if safePickle:
                # __getstate__ and __setstate__ of compiled_Qobj pass pointers
                # In 'safe' mod, these pointers are not used.
                if self.compiled == 1:
                    self.compiled_Qobj = CQobjCte()
                    self.compiled_Qobj.set_data(self.cte)
                elif self.compiled == 21:
                    self.compiled_Qobj = \
                        CQobjCteDense.__new__(CQobjCteDense)
                    self.compiled_Qobj.__setstate__(state[1])
                elif self.compiled == 31:
                    self.compiled_Qobj = CQobjCteOmp()
                    self.compiled_Qobj.set_data(self.cte)
                    self.compiled_Qobj.set_threads(self.omp)
                elif self.compiled in (2, 3):
                    self.compiled_Qobj = CQobjEvoTd()
                    self.compiled_Qobj.set_data(self.cte, self.ops)
                elif self.compiled in (22, 23):
                    self.compiled_Qobj = \
                        CQobjEvoTdDense.__new__(CQobjEvoTdDense)
                    self.compiled_Qobj.__setstate__(state[1])
                elif self.compiled in (12, 13):
                    self.compiled_Qobj = \
                        CQobjEvoTdMatched.__new__(CQobjEvoTdMatched)
                    self.compiled_Qobj.__setstate__(state[1])
                elif self.compiled in (32, 33):
                    self.compiled_Qobj = CQobjEvoTdOmp()
                    self.compiled_Qobj.set_data(self.cte, self.ops)
                    self.compiled_Qobj.set_threads(self.omp)
                elif self.compiled in (42, 43):
                    self.compiled_Qobj = \
                        CQobjEvoTdMatchedOmp.__new__(CQobjEvoTdMatchedOmp)
                    self.compiled_Qobj.__setstate__(state[1])
                    self.compiled_Qobj.set_threads(self.omp)
                if self.compiled % 10 == 3:
                    self.compiled_Qobj.set_factor(obj=self.coeff_get)
                elif self.compiled % 10 == 2:
                    self.compiled_Qobj.set_factor(func=self.coeff_get)
            else:
                if self.compiled == 1:
                    self.compiled_Qobj = CQobjCte.__new__(CQobjCte)
                elif self.compiled == 21:
                    self.compiled_Qobj = \
                        CQobjCteDense.__new__(CQobjCteDense)
                elif self.compiled == 31:
                    self.compiled_Qobj = \
                        CQobjCteOmp.__new__(CQobjCteOmp)
                    self.compiled_Qobj.set_threads(self.omp)
                elif self.compiled in (2, 3):
                    self.compiled_Qobj = CQobjEvoTd.__new__(CQobjEvoTd)
                elif self.compiled in (22, 23):
                    self.compiled_Qobj = \
                        CQobjEvoTdDense.__new__(CQobjEvoTdDense)
                elif self.compiled in (12, 13):
                    self.compiled_Qobj = \
                        CQobjEvoTdMatched.__new__(CQobjEvoTdMatched)
                elif self.compiled in (32, 33):
                    self.compiled_Qobj = CQobjEvoTdOmp.__new__(CQobjEvoTdOmp)
                    self.compiled_Qobj.set_threads(self.omp)
                elif self.compiled in (42, 43):
                    self.compiled_Qobj = \
                        CQobjEvoTdMatchedOmp.__new__(CQobjEvoTdMatchedOmp)
                    self.compiled_Qobj.set_threads(self.omp)
                self.compiled_Qobj.__setstate__(state[1])


# Function defined inside another function cannot be pickled,
# Using class instead
class _UnitedFuncCaller:
    def __init__(self, funclist, args):
        self.funclist = funclist
        self.args = args

    def set_args(self, args):
        self.args = args

    def __call__(self, t, args=None):
        if args:
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        out = []
        for func in self.funclist:
            out.append(func(t, now_args))
        return out


class _Norm2():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return self.func(t, args)*np.conj(self.func(t, args))


class _Conj():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return np.conj(self.func(t, args))


class _Prod():
    def __init__(self, f, g):
        self.func_1 = f
        self.func_2 = g

    def __call__(self, t, args):
        return self.func_1(t, args)*self.func_2(t, args)


class _Add():
    def __init__(self, fs):
        self.funcs = fs

    def __call__(self, t, args):
        return np.sum([f(t, args) for f in self.funcs])

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

import functools
import numbers
import os
import re
import sys
from types import FunctionType, BuiltinFunctionType

import numpy as np
import scipy
from scipy.interpolate import CubicSpline, interp1d

from .qobj import Qobj
from .cy.spmatfuncs import (
    cy_expect_rho_vec, cy_expect_psi, spmv,
)
from .cy.cqobjevo import CQobjEvo
from .coefficient import coefficient

from .. import settings as qset
if qset.has_openmp:
    from .cy.openmp.cqobjevo_omp import (
        CQobjCteOmp, CQobjEvoTdOmp, CQobjEvoTdMatchedOmp,
    )

from . import data as _data

try:
    import cython
    use_cython = [True]
    del cython
except ImportError:
    use_cython = [False]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# object for each time dependent element of the QobjEvo
# qobj : the Qobj of element ([*Qobj*, f])
# get_coeff : a callable that take (t, args) and return the coeff at that t
# coeff : The coeff as a string, array or function as provided by the user.
# type : flag for the type of coeff
class EvoElement:
    def __init__(self, qobj, coeff):
        self.qobj = qobj
        self.coeff = coeff


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
        def f1_t(t, args):
            return np.exp(-1j * t * args["w1"])

        def f2_t(t, args):
            return np.cos(t * args["w2"])

        H = QobjEvo([H0, [H1, f1_t], [H2, f2_t]], args={"w1":1., "w2":2.})

    For string based coeffients, the string must be a compilable python code
    resulting in a complex. The following symbols are defined:
        sin cos tan asin acos atan pi
        sinh cosh tanh asinh acosh atanh
        exp log log10 erf zerf sqrt
        real imag conj abs norm arg proj
        numpy as np, and scipy.special as spe.
    *Examples*
        H = QobjEvo([H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
                    args={"w1":1.,"w2":2.})

    For numpy array format, the array must be an 1d of dtype float or complex.
    A list of times (float64) at which the coeffients must be given (tlist).
    The coeffients array must have the same len as the tlist.
    The time of the tlist do not need to be equidistant, but must be sorted.
    By default, a cubic spline interpolation will be used for the coefficient
    at time t.
    If the coefficients are to be treated as step function, use the arguments
    args = {"_step_func_coeff": True}
    *Examples*
        tlist = np.logspace(-5,0,100)
        H = QobjEvo([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)

    args is a dict of (name:object). The name must be a valid variables string.
    Some solvers support arguments that update at each call:
    sesolve, mesolve, mcsolve:
        state can be obtained with:
            "state_vec":psi0, args["state_vec"] = state as 1D np.ndarray
            "state_mat":psi0, args["state_mat"] = state as 2D np.ndarray
            "state":psi0, args["state"] = state as Qobj

            This Qobj is the initial value.

        expectation values:
            "expect_op_n":0, args["expect_op_n"] = expect(e_ops[int(n)], state)
            expect is <phi|O|psi> or tr(state * O) depending on state
            dimensions

    mcsolve:
        collapse can be obtained with:
            "collapse":list => args[name] == list of collapse
            each collapse will be appended to the list as (time, which c_ops)

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

    ops : list of EvoElement
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

    dynamics_args : list
        arguments that change during evolution

    tlist : array_like
        List of times at which the numpy-array coefficients are applied.

    compiled : string
        Has the cython version of the QobjEvo been created

    compiled_qobjevo : cy_qobj (CQobjCte or CQobjEvoTd)
        Cython version of the QobjEvo

    coeff_get : callable object
        object called to obtain a list of coefficient at t

    coeff_files : list
        runtime created files to delete with the instance

    dummy_cte : bool
        is self.cte a empty Qobj

    const : bool
        Indicates if quantum object is Constant

    type : string
        information about the type of coefficient
            "string", "func", "array",
            "spline", "mixed_callable", "mixed_compilable"

    num_obj : int
        number of Qobj in the QobjEvo : len(ops) + (1 if not dummy_cte)

    use_cython : bool
        flag to compile string to cython or python


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

    _cdc()
        Return self.dag() * self.

    permute(order)
        Returns composite qobj with indices reordered.

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

    __call__(t, data=False, state=None, args={}):
        Return the Qobj at time t.
        *Faster after compilation

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
    def __init__(self, Q_object=[], args={}, copy=True,
                 tlist=None, state0=None, e_ops=[]):
        if isinstance(Q_object, QobjEvo):
            if copy:
                self._inplace_copy(Q_object)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args)
                for i, dargs in enumerate(self.dynamics_args):
                    e_int = dargs[1] == "expect" and isinstance(dargs[2], int)
                    if e_ops and e_int:
                        self.dynamics_args[i] = (dargs[0], "expect",
                                                 e_ops[dargs[2]])
                if state0 is not None:
                    self._dynamics_args_update(0., state0)
            return

        self.const = False
        self.dummy_cte = False
        self.args = args.copy()
        self.dynamics_args = []
        self.cte = None
        self.tlist = tlist
        self.compiled = ""
        self.compiled_qobjevo = None
        self.type = "none"
        self.omp = 0
        self.coeff_files = []
        self.use_cython = use_cython[0]

        if isinstance(Q_object, list) and len(Q_object) == 2:
            if isinstance(Q_object[0], Qobj) and not isinstance(Q_object[1],
                                                                (Qobj, list)):
                # The format is [Qobj, f/str]
                Q_object = [Q_object]

        self.ops = []

        if isinstance(Q_object, Qobj):
            self.cte = Q_object
            self.const = True
            self.type = "cte"
        else:
            try:
                use_step_func = self.args["_step_func_coeff"]
            except KeyError:
                use_step_func = 0

            for op in Q_object:
                if isinstance(op, Qobj):
                    if self.cte is None:
                        self.cte = op
                    else:
                        self.cte += op
                else:
                    self.ops.append(
                        EvoElement(op[0],
                                   coefficient(
                                        op[1],
                                        tlist=tlist,
                                        args=args,
                                        _stepInterpolation=use_step_func
                        )))

            if self.cte is None:
                self.cte = self.ops[0].qobj * 0
                self.dummy_cte = True

            try:
                cte_copy = self.cte.copy()
                # test is all qobj are compatible (shape, dims)
                for op in self.ops:
                    cte_copy += op.qobj
            except Exception as e:
                raise TypeError("Qobj not compatible.") from e

            if not self.ops:
                self.const = True

        self._args_checks()
        if e_ops:
            for i, dargs in enumerate(self.dynamics_args):
                if dargs[1] == "expect" and isinstance(dargs[2], int):
                    self.dynamics_args[i] = (dargs[0], "expect",
                                             QobjEvo(e_ops[dargs[2]]))
        if state0 is not None:
            self._dynamics_args_update(0., state0)

    def _args_checks(self):
        # TODO: all dynamic_arguments to be moved elsewhere
        return

    def __call__(self, t, data=False, state=None, args={}):
        try:
            t = float(t)
        except Exception as e:
            raise TypeError("t should be a real scalar.") from e

        if state is not None:
            self._dynamics_args_update(t, state)

        if args:
            if not isinstance(args, dict):
                raise TypeError("The new args must be in a dict")
            old_args = self.args.copy()
            self.args.update(args)
            self.arguments(self.args)
            op_t = self.__call__(t, data=data)
            self.args = old_args
            self.arguments(self.args)
        elif self.const:
            if data:
                op_t = self.cte.data.copy()
            else:
                op_t = self.cte.copy()
        elif self.compiled:
            op_t = self.compiled_qobjevo.call(t, data=data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part.qobj.data * part.coeff(t)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part.qobj * part.coeff(t)

        return op_t

    def _dynamics_args_update(self, t, state):
        # TODO: sort this out.
        if isinstance(state, _data.Data):
            state = state.to_array()

        if isinstance(state, Qobj):
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state.full().ravel("F")
                elif what == "mat":
                    self.args[name] = state.full()
                elif what == "Qobj":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)

        elif isinstance(state, np.ndarray) and state.ndim == 1:
            s1 = self.cte.shape[1]
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)
                elif state.shape[0] == s1 and self.cte.issuper:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=self.cte.dims[1])
                elif state.shape[0] == s1:
                    mat = state.reshape((-1, 1))
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat,
                                               dims=[self.cte.dims[1], [1]])
                elif state.shape[0] == s1*s1:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=[self.cte.dims[1],
                                                          self.cte.dims[1]])

        elif isinstance(state, np.ndarray) and state.ndim == 2:
            s1 = self.cte.shape[1]
            new_l = int(np.sqrt(s1))
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state.ravel("F")
                elif what == "mat":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)
                elif state.shape[1] == 1:
                    self.args[name] = Qobj(state, dims=[self.cte.dims[1],[1]])
                elif state.shape[1] == s1:
                    self.args[name] = Qobj(state, dims=self.cte.dims)
                else:
                    self.args[name] = Qobj(state)

        else:
            raise TypeError("state must be a Qobj or np.ndarray")

    @property
    def num_obj(self):
        return (len(self.ops) if self.dummy_cte else len(self.ops) + 1)

    def copy(self):
        new = QobjEvo(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.dynamics_args = self.dynamics_args.copy()
        new.tlist = self.tlist
        new.dummy_cte = self.dummy_cte
        new.type = self.type
        new.compiled = False
        new.compiled_qobjevo = None
        new.coeff_files = []
        new.use_cython = self.use_cython

        for op in self.ops:
            new.ops.append(EvoElement(op.qobj.copy(), op.coeff.copy()))

        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.dynamics_args = other.dynamics_args
        self.tlist = other.tlist
        self.dummy_cte = other.dummy_cte
        self.type = other.type
        self.compiled = ""
        self.compiled_qobjevo = None
        self.ops = []
        self.coeff_files = []
        self.use_cython = other.use_cython

        for op in other.ops:
            self.ops.append(EvoElement(op.qobj.copy(), op.coeff.copy()))

    def arguments(self, new_args):
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(new_args)
        for op in self.ops:
            op.coeff.arguments(self.args)
        return

    def solver_set_args(self, new_args, state, e_ops):
        self.arguments(new_args)
        return
        # TODO: all this is to be moved in a new object in solver
        self.dynamics_args = []
        self.args.update(new_args)
        self._args_checks()
        for i, dargs in enumerate(self.dynamics_args):
            if dargs[1] == "expect" and isinstance(dargs[2], int):
                self.dynamics_args[i] = (dargs[0], "expect",
                                         QobjEvo(e_ops[dargs[2]]))
                if self.compiled:
                    self.dynamics_args[i][2].compile()
        self._dynamics_args_update(0., state)
        if self.compiled and self.compiled.split()[2] != "cte":
            if isinstance(self.coeff_get, StrCoeff):
                self.coeff_get.set_args(self.args)
                self.coeff_get._set_dyn_args(self.dynamics_args)
            elif isinstance(self.coeff_get, _UnitedFuncCaller):
                self.coeff_get.set_args(self.args, self.dynamics_args)

    def to_list(self):
        list_qobj = []
        if not self.dummy_cte:
            list_qobj.append(self.cte)
        for op in self.ops:
            list_qobj.append([op.qobj, op.coeff])
        return list_qobj

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
                self.ops.append(EvoElement(op.qobj.copy(), op.coeff.copy()))
                l += 1
            self.args.update(**other.args)
            self.dynamics_args += other.dynamics_args
            self.const = self.const and other.const
            self.dummy_cte = self.dummy_cte and other.dummy_cte
            self.compiled = ""
            self.compiled_qobjevo = None
        else:
            self.cte += other
            self.dummy_cte = False
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
                op.qobj = other * op.qobj
            return res
        else:
            res *= other
            return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, numbers.Number):
            self.cte *= other
            for op in self.ops:
                op.qobj *= other
        elif isinstance(other, QobjEvo):
            if other.const:
                self.cte *= other.cte
                for op in self.ops:
                    op.qobj *= other.cte
            elif self.const:
                cte = self.cte.copy()
                self = other.copy()
                self.cte = cte * self.cte
                for op in self.ops:
                    op.qobj = cte*op.qobj
            else:
                cte = self.cte.copy()
                self.cte *= other.cte
                new_terms = []
                old_ops = self.ops
                if not other.dummy_cte:
                    for op in old_ops:
                        new_terms.append(EvoElement(op.qobj * other.cte,
                                                    op.coeff))
                if not self.dummy_cte:
                    for op in other.ops:
                        new_terms.append(EvoElement(cte * op.qobj,
                                                    op.coeff))
                for op_L in old_ops:
                    for op_R in other.ops:
                        new_terms.append(EvoElement(op_L.qobj * op_R.qobj,
                                                    op_L.coeff * op_R.coeff))
                self.ops = new_terms
                self.args.update(other.args)
                self.dynamics_args += other.dynamics_args
                self.dummy_cte = self.dummy_cte and other.dummy_cte

        else:
            raise TypeError("QobjEvo can only be multiplied"
                            " with QobjEvo, Qobj or numbers")
        return self

    def __div__(self, other):
        if isinstance(other, numbers.Number):
            res = self.copy()
            res *= 1 / complex(other)
            return res
        raise TypeError('Incompatible object for division')

    def __idiv__(self, other):
        if not isinstance(other, numbers.Number):
            raise TypeError('Incompatible object for division')
        self *= 1 / complex(other)
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        res = self.copy()
        res.cte = -res.cte
        for op in res.ops:
            op.qobj = -op.qobj
        return res

    # Transformations
    def trans(self):
        res = self.copy()
        res.cte = res.cte.trans()
        for op in res.ops:
            op.qobj = op.qobj.trans()
        return res

    def conj(self):
        res = self.copy()
        res.cte = res.cte.conj()
        res.ops = [EvoElement(op.qobj.conj(), op.coeff.conj())
                   for op in self.ops]
        return res

    def dag(self):
        res = self.copy()
        res.cte = res.cte.dag()
        res.ops = [EvoElement(op.qobj.dag(), op.coeff.conj())
                   for op in self.ops]
        return res

    def _cdc(self):
        """return a.dag * a """
        if not self.num_obj == 1:
            res = self.dag()
            res *= self
        else:
            res = self.copy()
            res.cte = res.cte.dag() * res.cte
            res.ops = [EvoElement(op.qobj.dag() * op.qobj, op.coeff._cdc())
                       for op in self.ops]
        return res

    def _shift(self):
        self.compiled = ""
        self.args.update({"_t0": 0})
        self.ops = [EvoElement(op.qobj, op.coeff._shift()) for op in self.ops]
        return self

    # Unitary function of Qobj
    def tidyup(self, atol=1e-12):
        self.cte = self.cte.tidyup(atol)
        for op in self.ops:
            op.qobj = op.qobj.tidyup(atol)
        return self

    def _compress_make_set(self):
        sets = []
        callable_flags = ["func", "spline"]
        for i, op1 in enumerate(self.ops):
            already_matched = any(i in _set for _set in sets)
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.qobj == op2.qobj:
                        this_set.append(j+i+1)
                sets.append(this_set)

        fsets = []
        for i, op1 in enumerate(self.ops):
            already_matched = any(i in _set for _set in fsets)
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.coeff is op2.coeff:
                        this_set.append(j+i+1)
                fsets.append(this_set)
        return sets, fsets

    def _compress_merge_qobj(self, sets):
        callable_flags = ["func", "spline"]
        new_ops = []
        for _set in sets:
            if len(_set) == 1:
                new_ops.append(self.ops[_set[0]])
            else:
                # Would be nice but need to defined addition with 0 (int)
                # new_coeff = sum(self.ops[i].coeff for i in _set)
                new_coeff = self.ops[_set[0]].coeff
                for i in _set[1:]:
                    new_coeff += self.ops[i].coeff
                new_ops.append(EvoElement(self.ops[_set[0]].qobj, new_coeff))

        self.ops = new_ops

    def _compress_merge_func(self, fsets):
        new_ops = []
        for _set in fsets:
            base = self.ops[_set[0]]
            new_op = [None, base.coeff]
            if len(_set) == 1:
                new_op[0] = base.qobj
            else:
                new_op[0] = base.qobj.copy()
                for i in _set[1:]:
                    new_op[0] += self.ops[i].qobj
            new_ops.append(EvoElement(*new_op))
        self.ops = new_ops

    def compress(self):
        self.tidyup()
        sets, fsets = self._compress_make_set()
        N_sets = len(sets)
        N_fsets = len(fsets)
        num_ops = len(self.ops)
        self.compiled = ""
        self.compiled_qobjevo = None

        if N_sets < num_ops and N_fsets < num_ops:
            # Both could be better
            if N_sets < N_fsets:
                self._compress_merge_qobj(sets)
            else:
                self._compress_merge_func(fsets)
            sets, fsets = self._compress_make_set()
            N_sets = len(sets)
            N_fsets = len(fsets)
            num_ops = len(self.ops)

        if N_sets < num_ops:
            self._compress_merge_qobj(sets)
        elif N_fsets < num_ops:
            self._compress_merge_func(fsets)

    def permute(self, order):
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op.qobj = op.qobj.permute(order)
        return res

    # function to apply custom transformations
    def apply(self, function, *args, **kw_args):
        self.compiled = ""
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op.qobj = function(op.qobj, *args, **kw_args)
        return res

    def expect(self, t, state, herm=0):
        if not isinstance(t, (int, float)):
            raise TypeError("The time need to be a real scalar")
        if isinstance(state, Qobj):
            if self.cte.dims[1] == state.dims[0]:
                vec = state.data
            elif self.cte.dims[1] == state.dims:
                vec = state.data
            else:
                raise Exception("Dimensions do not fit")
        elif isinstance(state, np.ndarray):
            vec = _data.create(state)
        else:
            raise TypeError("The vector must be an array or Qobj")

        if vec.shape[0]*vec.shape[1] == self.cte.shape[1]:
            if self.compiled:
                exp = self.compiled_qobjevo.expect(t, vec)
            elif self.cte.issuper:
                self._dynamics_args_update(t, state)
                exp = cy_expect_rho_vec(
                    self.__call__(t, data=True).as_scipy(),
                    vec.to_array().reshape(-1), 0)
            else:
                self._dynamics_args_update(t, state)
                exp = cy_expect_psi(self.__call__(t, data=True).as_scipy(),
                                    vec.to_array()[:, 0], 0)
        elif vec.shape[0]*vec.shape[1] == self.cte.shape[1]**2:
            if self.compiled:
                print("here2")
                exp = self.compiled_qobjevo.expect(t, vec)
            else:
                self._dynamics_args_update(t, state)
                exp = (self.__call__(t, data=True) *
                       vec.to_array().reshape((self.cte.shape[1],
                                               self.cte.shape[1])).T).trace()
        else:
            raise Exception("The shapes do not match")

        if herm:
            return exp.real
        else:
            return exp

    def mul_vec(self, t, vec):
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(vec, Qobj):
            if self.cte.dims[1] != vec.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = vec.dims
            vec = _data.dense.fast_from_numpy(vec.full())
        elif isinstance(vec, np.ndarray):
            if vec.ndim != 1:
                raise Exception("The vector must be 1d")
            # TODO: do this properly.
            vec = _data.Dense(vec[:, None])
        else:
            raise TypeError("The vector must be an array or Qobj")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        if self.compiled:
            out = self.compiled_qobjevo.matmul(t, vec).as_ndarray()[:, 0]
        else:
            self._dynamics_args_update(t, vec)
            out = spmv(self.__call__(t, data=True).as_scipy(),
                       np.ascontiguousarray(vec.to_array()[:, 0]))

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def mul_mat(self, t, mat):
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(mat, Qobj):
            if self.cte.dims[1] != mat.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = mat.dims
            mat = _data.dense.fast_from_numpy(mat.full())
        elif isinstance(mat, np.ndarray):
            if mat.ndim != 2:
                raise Exception("The matrice must be 2d")
            mat = _data.Dense(mat)
        else:
            raise TypeError("The vector must be an array or Qobj")
        if mat.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        if self.compiled:
            out = self.compiled_qobjevo.matmul(t, mat).as_ndarray()
        else:
            self._dynamics_args_update(t, mat)
            out = self.__call__(t, data=True).to_array() @ mat.as_ndarray()

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def compile(self, code=False, matched=False, dense=False, omp=0):
        # TODO: remove: always compiled
        # string compilation is now done at initialisation
        self.tidyup()
        # matched and dense useless, to remove dense
        # OMP also to move somewhere else (OPM_NUM_THREADS)
        # All ignored.
        # All dyn_args ignored
        self.tidyup()
        if self.compiled:
            return
        self.compiled_qobjevo = CQobjEvo(self.cte, self.ops)
        self.compiled = True

    def _get_coeff(self, t):
        out = []
        for part in self.ops:
            out.append(part.coeff(t))
        return out

    def coeff_get(self, t):
        # TODO: remove once no longer used
        out = []
        for part in self.ops:
            out.append(part.coeff(t))
        return out

    def __getstate__(self):
        _dict_ = self.__dict__.copy()
        # TODO: get rid of the separate CQobjEvo for OpenMP or make it
        # pickleable.  The new (regular) CQobjEvo is pickleable itself.
        # if "omp" in self.compiled:
        #     del _dict_['compiled_qobjevo']
        return _dict_

    def __setstate__(self, state):
        self.__dict__ = state
        """
        if "omp" in self.compiled:
            mat_type, _, td = self.compiled.split()
            if mat_type == "csr":
                if td == "cte":
                    self.compiled_qobjevo = CQobjCteOmp()
                    self.compiled_qobjevo.set_data(self.cte)
                    self.compiled_qobjevo.set_threads(self.omp)
                else:
                    self.compiled_qobjevo = CQobjEvoTdOmp()
                    self.compiled_qobjevo.set_data(self.cte, self.ops)
                    self.compiled_qobjevo.set_threads(self.omp)
            elif mat_type == "matched":
                self.compiled_qobjevo = \
                    CQobjEvoTdMatchedOmp.__new__(CQobjEvoTdMatchedOmp)
                self.compiled_qobjevo.set_threads(self.omp)
                self.compiled_qobjevo.__setstate__(state[1])
        """

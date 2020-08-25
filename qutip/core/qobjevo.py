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
from types import FunctionType, BuiltinFunctionType

import numpy as np
import scipy
from scipy.interpolate import CubicSpline, interp1d

import qutip
from .qobj import Qobj
from .cy.cqobjevo import CQobjEvo
from .coefficient import coefficient, CompilationOptions
from .cy.coefficient import Coefficient
from .superoperator import stack_columns, unstack_columns
from . import data as _data

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# object for each time dependent element of the QobjEvo
# qobj : the Qobj of element ([*Qobj*, f])
# coeff : The coeff as a string, array or function as provided by the user.
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
        List of times at which the numpy-array coefficients are applied.

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

    compiled_qobjevo : cQobjEvo
        Cython version of the QobjEvo

    dummy_cte : bool
        is self.cte a empty Qobj

    const : bool
        Indicates if quantum object is Constant

    num_obj : int
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

    _cdc()
        Return self.dag() * self.

    permute(order)
        Returns composite qobj with indices reordered.

    apply(f, *args, **kw_args)
        Apply the function f to every Qobj. f(Qobj) -> Qobj
        Return a modified QobjEvo and let the original one untouched

    tidyup(atol=1e-12)
        Removes small elements from quantum object.

    compress():
        Merge ops which are based on the same quantum object and coeff type.

    __call__(t, data=False, state=None, args={}):
        Return the Qobj at time t.

    mul(t, mat):
        Product of this at t time with the dense matrix mat.

    expect(t, psi, herm=False):
        Calculates the expectation value for the quantum object (if operator,
            no check) and state psi.
        Return only the real part if herm.

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
        self.compiled_qobjevo = None
        self.ops = []
        self._tlist = tlist

        if isinstance(Q_object, list) and len(Q_object) == 2:
            if isinstance(Q_object[0], Qobj) and not isinstance(Q_object[1],
                                                                (Qobj, list)):
                # The format is [Qobj, f/str]
                Q_object = [Q_object]

        if isinstance(Q_object, Qobj):
            self.cte = Q_object
            self.const = True
        else:
            use_step_func = self.args.get("_step_func_coeff", 0)
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
                                        _stepInterpolation=use_step_func,
                                        compile_opt=CompilationOptions(extra_import=" ")
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

        self._compile()

    def _args_checks(self):
        statedims = [self.cte.dims[1], [1]]
        for key in self.args:
            if key == "state" or key == "state_qobj":
                self.dynamics_args += [(key, "Qobj", None)]
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims)

            if key == "state_mat":
                self.dynamics_args += [("state_mat", "mat", None)]
                if isinstance(self.args[key], Qobj):
                    self.args[key] = self.args[key].full()
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims).full()

            if key == "state_vec":
                self.dynamics_args += [("state_vec", "vec", None)]
                if isinstance(self.args[key], Qobj):
                    self.args[key] = self.args[key].full().ravel("F")
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims).full().ravel("F")

            if key.startswith("expect_op_"):
                e_op_num = int(key[10:])
                self.dynamics_args += [(key, "expect", e_op_num)]
                self.args[key] = 0

    @property
    def shape(self):
        return self.cte.shape

    @property
    def dims(self):
        return self.cte.dims

    @property
    def tlist(self):
        from warnings import warn
        warn("tlist is to be removed from QobjEvo", DeprecationWarning)
        return self._tlist

    @tlist.setter
    def tlist(self, new):
        from warnings import warn
        warn("tlist is to be removed from QobjEvo", DeprecationWarning)
        self._tlist = new

    def _check_old_with_state(self):
        # Todo: remove, add deprecationwarning in 4.6.0
        pass

    def __call__(self, t, data=False, state=None, args={}):
        # TODO: no more data option. Always Qobj
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
        else:
            op_t = self.compiled_qobjevo.call(t, data=data)

        return op_t

    def _dynamics_args_update(self, t, state: _data.Data):
        for name, what, e_op in self.dynamics_args:
            self.args[name] = dynamic_argument(t, self.cte, state, what, e_op)

    @property
    def num_obj(self):
        return (len(self.ops) if self.dummy_cte else len(self.ops) + 1)

    def copy(self):
        new = QobjEvo(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.dynamics_args = self.dynamics_args.copy()
        new.dummy_cte = self.dummy_cte
        new.compiled_qobjevo = None
        new._tlist = self._tlist

        for op in self.ops:
            new.ops.append(EvoElement(op.qobj.copy(), op.coeff.copy()))
        new._compile()
        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.dynamics_args = other.dynamics_args
        self.dummy_cte = other.dummy_cte
        self.compiled_qobjevo = None
        self.ops = []
        self._tlist = other._tlist

        for op in other.ops:
            self.ops.append(EvoElement(op.qobj.copy(), op.coeff.copy()))
        self._compile()

    def arguments(self, new_args):
        """
        Update args.
        """
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(new_args)
        for op in self.ops:
            op.coeff.arguments(self.args)
        return

    def solver_set_args(self, new_args, state: _data.Data, e_ops):
        """
        Update args.
        """
        # TODO: remove when moving out dynamic_argument
        if not isinstance(state, _data.Data):
            raise TypeError("state should be a data-layer object")
        self.dynamics_args = []
        self.args.update(new_args)
        self._args_checks()
        for i, dargs in enumerate(self.dynamics_args):
            if dargs[1] == "expect" and isinstance(dargs[2], int):
                self.dynamics_args[i] = (dargs[0], "expect",
                                         QobjEvo(e_ops[dargs[2]]))
        self._dynamics_args_update(0., state)
        for op in self.ops:
            op.coeff.arguments(self.args)
        self._compile()

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
            self.compiled_qobjevo = None
        else:
            self.cte += other
            self.dummy_cte = False
        self._compile()
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

    def __matmul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmatmul__(self, other):
        res = self.copy()
        if isinstance(other, Qobj):
            res.cte = other @ res.cte
            for op in res.ops:
                op.qobj = other @ op.qobj
            res._compile()
            return res
        else:
            res *= other
            return res

    def __imatmul__(self, other):
        res = self.copy()
        res *= other
        return res

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
            res._compile()
            return res
        else:
            res *= other
            return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, numbers.Number):
            self.cte *= other
            for op in self.ops:
                op.qobj *= other
        elif isinstance(other, Coefficient):
            for op in self.ops:
                op.coeff = op.coeff * other
            if not self.dummy_cte:
                self.ops.append(EvoElement(self.cte, other))
                self.cte *= 0
                self.dummy_cte = True
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
        self._compile()
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
        self._compile()
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        res = self.copy()
        res.cte = -res.cte
        for op in res.ops:
            op.qobj = -op.qobj
        res._compile()
        return res

    def __and__(self, other):
        """
        Syntax shortcut for tensor:
        A & B ==> tensor(A, B)
        """
        return qutip.tensor(self, other)

    # Transformations
    def trans(self):
        """ Transpose of the quantum object
        """
        res = self.copy()
        res.cte = res.cte.trans()
        for op in res.ops:
            op.qobj = op.qobj.trans()
        res._compile()
        return res

    def conj(self):
        """ Conjugate of the quantum object
        """
        res = self.copy()
        res.cte = res.cte.conj()
        res.ops = [EvoElement(op.qobj.conj(), op.coeff.conj())
                   for op in self.ops]
        res._compile()
        return res

    def dag(self):
        """ Hermitian adjoint of the quantum object
        """
        res = self.copy()
        res.cte = res.cte.dag()
        res.ops = [EvoElement(op.qobj.dag(), op.coeff.conj())
                   for op in self.ops]
        res._compile()
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
        res._compile()
        return res

    def _shift(self):
        self.args.update({"_t0": 0})
        self.ops = [EvoElement(op.qobj, op.coeff._shift()) for op in self.ops]
        self._compile()
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
        new_ops = []
        for _set in sets:
            if len(_set) == 1:
                new_ops.append(self.ops[_set[0]])
            else:
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
        """
        Find redundant contribution and merge them
        """
        self.tidyup()
        sets, fsets = self._compress_make_set()
        N_sets = len(sets)
        N_fsets = len(fsets)
        num_ops = len(self.ops)
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
        self._compile()

    def permute(self, order):
        """
        Permute tensor subspaces of the quantum object
        """
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op.qobj = op.qobj.permute(order)
        return res

    # function to apply custom transformations
    def apply(self, function, *args, **kw_args):
        """
        Apply function to each Qobj contribution.
        """
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op.qobj = function(op.qobj, *args, **kw_args)
        res._compile()
        return res

    def expect(self, t, state, herm=0):
        """
        Expectation value of the operator quantum object at time t
        for the given state.
        """
        if not isinstance(t, numbers.Real):
            raise TypeError("time needs to be a real scalar")
        if isinstance(state, Qobj):
            state = _data.Dense(state.full())
        elif isinstance(state, np.ndarray):
            state = _data.dense.fast_from_numpy(state)
        # TODO: remove shim once dispatch available.
        elif isinstance(state, _data.CSR):
            state = _data.Dense(state.to_array())
        elif isinstance(state, _data.Dense):
            pass
        else:
            raise TypeError("The vector must be an array or Qobj")

        if self.cte.issuper:
            state = _data.column_stack_dense(state)
        exp = self.compiled_qobjevo.expect(t, state)
        return exp.real if herm else exp

    def mul_vec(self, t, vec):
        """
        Product of the operator quantum object at time t
        with the given vector state.
        """
        # TODO: mostly used in test to compare with the cqobjevo version.
        # __mul__ sufficient? remove?
        # Still used in mcsolve, remove later
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

            vec = _data.Dense(vec[:, None])
        else:
            raise TypeError("The vector must be an array or Qobj")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        out = self.compiled_qobjevo.matmul(t, vec).as_ndarray()[:, 0]

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def mul(self, t, mat):
        """
        Product of the operator quantum object at time t
        with the given matrix state.
        """
        was_Qobj = False
        was_vec = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(mat, Qobj):
            if self.cte.dims[1] != mat.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = mat.dims
            mat = _data.dense.fast_from_numpy(mat.full())
        elif isinstance(mat, np.ndarray):
            if mat.ndim == 1:
                # TODO: do this properly.
                mat = _data.Dense(mat[:, None])
                was_vec = True
            elif mat.ndim == 2:
                mat = _data.Dense(mat)
            else:
                raise Exception("The matrice must be 1d or 2d")
        else:
            raise TypeError("The vector must be an array or Qobj")
        if mat.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        out = self.compiled_qobjevo.matmul(t, mat).as_ndarray()

        if was_Qobj:
            return Qobj(out, dims=dims)
        elif was_vec:
            return out[:, 0]
        else:
            return out

    def _compile(self, code=False, matched=False, dense=False):
        self.tidyup()
        self.compiled_qobjevo = CQobjEvo(self.cte, self.ops)
        self.compiled_qobjevo.set_dyn_args(self.dynamics_args,
                                           self.args, self.cte)


def _dynamic_argument_raise(op, state):
    raise TypeError(
        "unknown shape for evolution, evolver type "
        + repr(op.type)
        + " shape "
        + repr(op.shape)
        + ", state shape "
        + repr(state.shape)
    )


def dynamic_argument(t, op, state, what, e_op):
    # Input `state` is either ndarray or data type (_not_ Qobj).  First unify
    # to a data-layer type, then build the object we're asked for.  In the
    # future, "vec" and "matrix" should be data-layer types themselves, not
    # ndarray, but for now we leave them as-is.
    if isinstance(state, np.ndarray):
        state = _data.dense.fast_from_numpy(state)
    if what == "vec":
        out = stack_columns(state)
        return (out.as_ndarray()[:, 0] if isinstance(out, _data.Dense)
                else out.to_array()[:, 0])
    # Otherwise we have to create a proper infer what type of `state` we've
    # been passed, based on the operator doing the evolution.
    if op.issuper:
        if state.shape[0] == op.shape[1]*op.shape[1]:
            # We're evolving a superoperator which has been column-stacked.
            state = unstack_columns(state, (op.shape[1],)*2)
            type = 'super'
        elif state.shape[0] == op.shape[1]:
            state_rows = int(np.sqrt(state.shape[0]))
            if state.shape[1] != 1 or state_rows**2 != state.shape[0]:
                _dynamic_argument_raise(op, state)
            # We're evolving an operator which has been column-stacked.
            state = unstack_columns(state, (state_rows,)*2)
            type = 'oper'
        elif state.shape[0]*state.shape[1] == op.shape[1]:
            # We're evolving an operator which is in the normal format.
            type = 'oper'
        else:
            _dynamic_argument_raise(op, state)
    else:
        if state.shape[0] == state.shape[1] == op.shape[0] == op.shape[1]:
            # Evolving density matrix in 2D format.
            type = 'oper'
        elif state.shape[0] == op.shape[1]*op.shape[1] and state.shape[1] == 1:
            # Evolving column-stacked density matrix with unitary.
            state = unstack_columns(state, (op.shape[1],)*2)
            type = 'oper'
        elif state.shape[0] == op.shape[1] and state.shape[1] == 1:
            # We're evolving a ket with an operator.
            type = 'ket'
        else:
            _dynamic_argument_raise(op, state)

    if what == "mat":
        return (state.as_ndarray() if isinstance(state, _data.Dense)
                else state.to_array())
    if what == "expect":
        return e_op.expect(t, state)
    if what == "Qobj":
        if type == 'super':
            dims = op.dims
        elif type == 'oper':
            dims = op.dims[1] if op.issuper else op.dims
        elif type == 'ket':
            dims = [op.dims[1], [1]*len(op.dims[1])]
        else:
            raise RuntimeError("internal logic error, type=" + repr(type))
        # TODO: allow arbitrary data-layer types.
        return Qobj(state.to_array(), dims=dims, type=type, copy=False)
    raise RuntimeError("unexpected what=" + repr(what))

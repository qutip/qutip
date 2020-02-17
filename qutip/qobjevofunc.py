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
"""Time-dependent Quantum Object (QobjEvo) wrapper class
for function returning Qobj.
"""
__all__ = ['QobjEvoFunc']

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
import qutip.settings as qset
from types import FunctionType, BuiltinFunctionType
import numpy as np
from qutip.cy.spmatfuncs import cy_expect_rho_vec, cy_expect_psi, spmv
from qutip.cy.cqobjevo import CQobjFunc
from qutip.superoperator import spre, spost, liouvillian, lindblad_dissipator

import pickle
import sys
import scipy
import os


class QobjEvoFunc(QobjEvo):
    """A class for representing time-dependent quantum objects,
    such as quantum operators and states from a function or callable.

    The QobjEvoFunc class is a representation of time-dependent Qutip quantum
    objects (Qobj). This class implements math operations :
        +,- : QobjEvo, Qobj
        * : Qobj, C-number
        / : C-number
    and some common linear operator/state operations. The QobjEvoFunc is
    constructed from a function that return a Qobj from time and extra
    arguments.

    The signature of the function must be one of
    - f(t)
    - f(t, args)
    - f(t, **kwargs)
    - f(t, state, args)  -- for backward compatibility, to be deprecated --

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
            dimensions.

    mcsolve:
        collapse can be obtained with:
            "collapse":list => args[name] == list of collapse
            each collapse will be appended to the list as (time, which c_ops)

    Parameters
    ----------
    QobjEvoFunc(Q_object=[], args={}, copy=True,
                 tlist=None, state=None, e_ops=[])

    Q_object : callable
        Function/method that return the Qobj at time t.

    args : dictionary that contain the arguments for coefficients.

    copy : bool
        If Q_object is already a QobjEvo, return a copy.

    tlist : array_like
        List of times at which the numpy-array coefficients are applied. Times
        must be equidistant and start from 0.

    state : Qobj
        First state to use if the state is used for args.

    e_ops : list of Qobj
        Operators from which args can be build.
        args["expect_op_0"] = expect(e_ops[0], state)

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

    compiled_qobjevo : cy_qobj (CQobjCte or CQobjEvoTd)
        Cython version of the QobjEvo

    dummy_cte : bool
        is self.cte a dummy Qobj

    const : bool
        Indicates if quantum object is Constant

    type : int
        information about the type of coefficient
            "string", "func", "array",
            "spline", "mixed_callable", "mixed_compilable"

    num_obj : int
        number of Qobj in the QobjEvo : len(ops) + (1 if not dummy_cte)


    Methods
    -------
    copy() :
        Create copy of QobjEvoFunc

    arguments(new_args, state, e_ops):
        Update the args and set the dynamics_args

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
        Only possible if num_obj == 1

    permute(order)
        Returns composite qobj with indices reordered.

    apply(f, *args, **kw_args)
        Apply the function f to every Qobj. f(Qobj) -> Qobj
        Return a modified QobjEvo and let the original one untouched

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
                 tlist=None, state=None, e_ops=[]):
        if isinstance(Q_object, QobjEvoFunc):
            if copy:
                self._inplace_copy(Q_object)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args, state, e_ops)
            return
        self.args = args.copy() if copy else args
        self.dynamics_args = []

        self.compiled = ""
        self.compiled_qobjevo = None
        self.operation_stack = []
        self.shifted = False
        self.const = False

        # Dummy attributes from QobjEvo
        self.coeff_get = None
        self.tlist = None
        self.omp = 0
        self.num_obj = 1
        self.dummy_cte = True
        self.type = ""

        if not callable(Q_object):
            raise TypeError("expected a function")
        self._args_checks(state, e_ops)

        self.func = set_signature(Q_object, self.args, state)
        # Let Q_object call raise an error if wrong definition
        self.cte = self.func(0, self.args)
        if not isinstance(self.cte, Qobj):
            raise TypeError("QobjEvoFunc require un function returning a Qobj")

    def __del__(self):
        pass

    def __call__(self, t, data=False, state=None, args={}):
        try:
            t = float(t)
        except Exception as e:
            raise TypeError("t should be a real scalar.") from e
        if state is not None:
            self._dynamics_args_update(t, state)
        out = self._get_qobj(t, args)
        if data:
            out = out.data
        return out

    def _get_qobj(self, t, args={}):
        if args:
            if not isinstance(args, dict):
                raise TypeError("The new args must be in a dict")
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        if self.shifted:
            t += args["_t0"]
        qobj = self.func(t, now_args)
        for transform in self.operation_stack:
            qobj = transform(qobj, t, now_args)
        return qobj

    def copy(self):
        new = QobjEvoFunc.__new__(QobjEvoFunc)
        new.__dict__ = self.__dict__.copy()
        new.args = self.args.copy()
        new.dynamics_args = self.dynamics_args.copy()
        new.operation_stack = [oper.copy() for oper in self.operation_stack]
        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.args = other.args.copy()
        self.dynamics_args = other.dynamics_args
        self.operation_stack = [oper.copy() for oper in other.operation_stack]
        self.type = other.type
        self.shifted = other.shifted
        self.compiled = ""
        self.compiled_qobjevo = None
        self.func = other.func

    def arguments(self, new_args, state=None, e_ops=[]):
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        # remove dynamics_args that are to be refreshed
        # self.dynamics_args = [dargs for dargs in self.dynamics_args
        #                       if dargs[0] not in new_args]
        self.args.update(new_args)
        # self._args_checks(state=state, e_ops=e_ops)
        for key in new_args:
            if isinstance(self.args[key], StateArgs):
                self.dynamics_args = [dargs for dargs in self.dynamics_args
                                      if dargs[0] != key]
                self.dynamics_args += [(key, *self.args[key]())]

    def to_list(self):
        return self.func

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
            self.operation_stack.append(_Block_Sum_Qoe(other))
        elif isinstance(other, Qobj):
            self.operation_stack.append(_Block_Sum(other))
        else:
            try:
                other = Qobj(other)
                self.operation_stack.append(_Block_Sum(other))
            except Exception:
                return NotImplemented
        if not self._reset_type():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
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
        if isinstance(other, QobjEvo):
            res.operation_stack.append(_Block_rmul_Qoe(other))
        else:
            res.operation_stack.append(_Block_rmul(other))
        if not self._reset_type():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
        return res

    def __imul__(self, other):
        if isinstance(other, QobjEvo):
            self.operation_stack.append(_Block_mul_Qoe(other))
        else:
            self.operation_stack.append(_Block_mul(other))
        if not self._reset_type():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
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
        res.operation_stack.append(_Block_neg())
        return res

    # Transformations
    def trans(self):
        res = self.copy()
        res.operation_stack.append(_Block_trans())
        return res

    def conj(self):
        res = self.copy()
        res.operation_stack.append(_Block_conj())
        return res

    def dag(self):
        res = self.copy()
        res.operation_stack.append(_Block_dag())
        return res

    def spre(self):
        res = self.copy()
        res.operation_stack.append(_Block_pre())
        self._reset_type()
        return res

    def spost(self):
        res = self.copy()
        res.operation_stack.append(_Block_post())
        self._reset_type()
        return res

    def liouvillian(self, c_ops=[], chi=None):
        res = self.copy()
        if res.cte.issuper: return res
        c_ops = [qobjevo_maker(c_op) for c_op in c_ops]
        res.operation_stack.append(_Block_liouvillian(c_ops, chi))
        self._reset_type()
        return res

    def lindblad_dissipator(self, chi=0):
        res = self.copy()
        chi = 0 if chi is None else chi
        if res.cte.issuper:
            return res * np.exp(1j * chi) if chi else res
        res.operation_stack.append(_Block_lindblad_dissipator(chi))
        self._reset_type()
        return res

    def _cdc(self):
        """return a.dag * a """
        res = self.copy()
        res.operation_stack.append(_Block_cdc())
        return res

    def _prespostdag(self):
        """return spre(a) * spost(a.dag()) """
        res = self.copy()
        res.operation_stack.append(_Block_prespostdag())
        return res

    def _liouvillian_h(self):
        """return 1j * (spre(a) - spost(a)) """
        res = self.copy()
        if res.cte.issuper: return res
        res.operation_stack.append(_Block_liouvillian_H())
        return res

    def _shift(self):
        """shift t by args("_t0") """
        res = self.copy()
        res.shifted = True
        res.args["_t0"] = 0
        return res

    # Unitary function of Qobj
    def tidyup(self, atol=1e-12):
        for transform in self.operation_stack:
            transform.tidyup(atol)
        return self

    def compress(self):
        pass

    def _reset_type(self):
        try:
            self.cte = self._get_qobj(0.)
        except Exception:
            return False
        return True

    def permute(self, order):
        res = self.copy()
        res.operation_stack.append(_Block_permute(order))
        return res

    # function to apply custom transformations
    def apply(self, function, *args, **kw_args):
        res = self.copy()
        res.operation_stack.append(_Block_apply(function, args, kw_args))
        self._reset_type()
        return res

    def expect(self, t, state, herm=0):
        if not isinstance(t, (int, float)):
            raise TypeError("The time need to be a real scalar")
        if isinstance(state, Qobj):
            if self.cte.dims[1] == state.dims[0]:
                vec = state.full().ravel("F")
            elif self.cte.dims[1] == state.dims:
                vec = state.full().ravel("F")
            else:
                raise Exception("Dimensions do not fit")
        elif isinstance(state, np.ndarray):
            vec = state.reshape((-1))
        else:
            raise TypeError("The vector must be an array or Qobj")

        if vec.shape[0] == self.cte.shape[1]:
            if self.cte.issuper:
                self._dynamics_args_update(t, state)
                exp = cy_expect_rho_vec(self.__call__(t, data=True), vec, 0)
            else:
                self._dynamics_args_update(t, state)
                exp = cy_expect_psi(self.__call__(t, data=True), vec, 0)
        elif vec.shape[0] == self.cte.shape[1]**2:
            self._dynamics_args_update(t, state)
            exp = (self.__call__(t, data=True) *
                   vec.reshape((self.cte.shape[1],
                                self.cte.shape[1]))).trace()
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
            vec = vec.full().ravel()
        elif not isinstance(vec, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if vec.ndim != 1:
            raise Exception("The vector must be 1d")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        self._dynamics_args_update(t, vec)
        out = spmv(self.__call__(t, data=True), vec)

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
            mat = mat.full()
        if not isinstance(mat, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if mat.ndim != 2:
            raise Exception("The matrice must be 2d")
        if mat.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        self._dynamics_args_update(t, mat)
        out = self.__call__(t, data=True) * mat

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def compile(self, code=False, matched=False, dense=False, omp=0):
        self.compiled_qobjevo = CQobjFunc(self)
        self.compiled = "True"

    def __getstate__(self):
        _dict_ = {key: self.__dict__[key]
                  for key in self.__dict__ if key is not "compiled_qobjevo"}
        return _dict_

    def __setstate__(self, state):
        self.__dict__ = state
        self.compiled_qobjevo = None
        if self.compiled:
            self.compiled_qobjevo = CQobjFunc(self)


class _Block_transform:
    def __init__(self, other=None):
        self.other = other

    def tidyup(self, atol):
        pass

    def copy(self):
        return self.__class__(self.other)

    def __call__(self, obj, t, args={}):
        return obj


class _Block_neg(_Block_transform):
    def __call__(self, obj, t, args={}):
        return -obj


class _Block_Sum(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj + self.other


class _Block_mul(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj * self.other


class _Block_rmul(_Block_transform):
    def __call__(self, obj, t, args={}):
        return self.other * obj


class _Block_Sum_Qoe(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj + self.other(t, args=args)


class _Block_mul_Qoe(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj * self.other(t, args=args)


class _Block_rmul_Qoe(_Block_transform):
    def __call__(self, obj, t, args={}):
        return self.other(t, args=args) * obj


class _Block_trans(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.trans()


class _Block_conj(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.conj()


class _Block_dag(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.dag()


class _Block_cdc(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.dag() * obj


class _Block_prespostdag(_Block_transform):
    def __call__(self, obj, t, args={}):
        return spre(obj) * spost(obj.dag())


class _Block_permute(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.permute(self.other)


class _Block_pre(_Block_transform):
    def __call__(self, obj, t, args={}):
        return spre(obj)


class _Block_post(_Block_transform):
    def __call__(self, obj, t, args={}):
        return spost(obj)


class _Block_liouvillian(_Block_transform):
    def __init__(self, other, chi):
        self.other = other
        self.chi = chi

    def __call__(self, obj, t, args={}):
        c_ops = [op(t, args) for op in self.other]
        return liouvillian(obj, c_ops, self.chi)


class _Block_liouvillian_H(_Block_transform):
    def __call__(self, obj, t, args={}):
        return -1.0j * (spre(obj) - spost(obj))


class _Block_lindblad_dissipator(_Block_transform):
    def __call__(self, obj, t, args={}):
        return lindblad_dissipator(obj, chi=self.other)


class _Block_apply(_Block_transform):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def copy(self):
        return _Block_apply(self.func, self.args, self.kwargs.copy())

    def __call__(self, obj, t, args={}):
        return self.func(obj, *self.args, **self.kwargs)


from qutip.qobjevo_maker import StateArgs, set_signature

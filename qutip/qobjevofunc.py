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
__all__ = ['QObjEvoFunc', 'qobjevo_maker']

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
import qutip.settings as qset
from types import FunctionType, BuiltinFunctionType
import numpy as np
from qutip.cy.spmatfuncs import cy_expect_rho_vec, cy_expect_psi, spmv
from qutip.cy.cqobjevo import CQobjFunc

import pickle
import sys
import scipy
import os

def qobjevo_maker(Q_object=None, args={}, tlist=None, copy=True):
    if isinstance(Q_object, QobjEvo):
        return Q_object
    try:
        obj = QobjEvo(Q_object=None, args={}, tlist=None, copy=True)
    except Exception as e:
        obj = QObjEvoFunc(Q_object=None, args={}, tlist=None, copy=True)
    return obj


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, coeff_func):
        self.coeff_func = coeff_func

    def __call__(self, t, args={}):
        return self.coeff_func(t, args["_state_vec"], args)


class QObjEvoFunc(QobjEvo):
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
    *Examples*
        tlist = np.logspace(-5,0,100)
        H = QobjEvo([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)

    args is a dict of (name:object). The name must be a valid variables string.
    Some solvers support arguments that update at each call:
    sesolve, mesolve, mcsolve:
        state can be obtained with:
            name+"=vec":Qobj  => args[name] == state as 1D np.ndarray
            name+"=mat":Qobj  => args[name] == state as 2D np.ndarray
            name+"=Qobj":Qobj => args[name] == state as Qobj

            This Qobj is the initial value.

        expectation values:
            name+"=expect":O (Qobj/QobjEvo)  => args[name] == expect(O, state)
            expect is <phi|O|psi> or tr(state * O) depending on state dimensions

    mcsolve:
        collapse can be obtained with:
            name+"=collapse":list => args[name] == list of collapse
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
        Only possible if num_obj == 1

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
    def __init__(self, Q_object=None, args={}, tlist=None, copy=True):
        if isinstance(Q_object, QObjEvoFunc):
            if copy:
                self._inplace_copy(Q_object)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args)
            return
        self.args = args.copy() if copy else args
        self.dynamics_args = []
        self._args_checks()
        self.compiled = ""
        self.compiled_qobjevo = None
        self.operation_stack = []

        # Dummy attributes from QobjEvo
        self.coeff_get = None
        self.tlist = None
        self.omp = 0
        self.num_obj = 1
        self.dummy_cte = True
        self.const = False
        self.type = ""

        if callable(Q_object):
            try:
                self.cte = Q_object(0, args)
                self.func = Q_object
            except TypeError as e:
                self.cte = Qobj()
        if not isinstance(self.cte, Qobj):
            raise TypeError("QObjEvoFunc require un function returning a Qobj")

    def _args_checks(self, update=False):
        to_remove = []
        to_add = {}
        for key in self.args:
            if "=" in key:
                name, what = key.split("=")
                if what in ["Qobj", "vec", "mat"]:
                    # state first, expect last
                    if not update:
                        self.dynamics_args = [(name, what, None)] + self.dynamics_args
                        if name not in self.args:
                            if isinstance(self.args[key], Qobj):
                                if what == "Qobj":
                                    to_add[name] = self.args[key]
                                elif what == "mat":
                                    to_add[name] = self.args[key].full()
                                else:
                                    to_add[name] = self.args[key].full().ravel("F")
                            else:
                                if what == "Qobj":
                                    to_add[name] = Qobj(dims=[self.cte.dims[1],[1]])
                                elif what == "mat":
                                    to_add[name] = np.zeros((self.cte.shape[1],1))
                                else:
                                    to_add[name] = np.zeros((self.cte.shape[1]))

                elif what == "expect":
                    if isinstance(self.args[key], QobjEvo):
                        expect_op = self.args[key]
                    else:
                        expect_op = QobjEvo(self.args[key], copy=False)
                    if update:
                        for ops in self.dynamics_args:
                            if ops[0] == name:
                                ops = (name, what, expect_op)
                    else:
                        self.dynamics_args += [(name, what, expect_op)]
                        if name not in self.args:
                            to_add[name] = 0.
                else:
                    raise Exception("Could not understand dynamics args: " +
                                    what + "\nSupported dynamics args: "
                                    "Qobj, csr, vec, mat, expect")
                to_remove.append(key)

        for key in to_remove:
            del self.args[key]

        self.args.update(to_add)

    def _check_old_with_state(self, state):
        self.cte = self.func(0, state, args)
        self.func = _StateAsArgs(self.func)
        self.dynamics_args += [("_state_vec", "vec", None)]

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
        return op_t

    def _dynamics_args_update(self, t, state):
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
                    mat = state.reshape((-1,1))
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=[self.cte.dims[1],[1]])
                elif state.shape[0] == s1*s1:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=[self.cte.dims[1], self.cte.dims[1]])

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

    def _get_qobj(self, t, args={}):
        if args:
            if not isinstance(args, dict):
                raise TypeError("The new args must be in a dict")
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        qobj = self.func(t, now_args)
        for transform in self.operation_stack:
            qobj = transform(qobj, t, now_args)
        return qobj

    def copy(self):
        new = QObjEvoFunc.__new__()
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
        self.compiled = ""
        self.compiled_qobjevo = None
        self.func = other.func

    def arguments(self, args):
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(args)
        self._args_checks(True)

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
            self.operation_stack.append(_Block_Sum_Qo(other))
        else:
            try:
                other = Qobj(other)
                self.operation_stack.append(_Block_Sum_Qo(other))
            except Exception:
                return NotImplemented
        if not self._reset_type():
            self.operation_stack = self.operation_stack[:-1]
            return NotImplemented
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
            return NotImplemented
        return res

    def __imul__(self, other):
        if isinstance(other, QobjEvo):
            self.operation_stack.append(_Block_mul_Qoe(other))
        else:
            self.operation_stack.append(_Block_mul(other))
        if not self._reset_type():
            self.operation_stack = self.operation_stack[:-1]
            return NotImplemented
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
        c_ops = [qobjevo_maker(c_op) for c_op in c_ops]
        res.operation_stack.append(_Block_liouvillian(c_ops, chi))
        self._reset_type()
        return res

    def lindblad_dissipator(self, chi=0):
        res = self.copy()
        chi = 0 is chi is None else chi
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
        self.compiled = ""
        res = self.copy()
        res.operation_stack.append(_Block_apply(function, args, kw_args))
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
        self.other = None

    def tidyup(self, atol):
        pass

    def copy(self):
        return self.__class__(self.other)

    def __call__(self, obj, t, args={}):
        return obj


class _Block_neg(_Block_transform):
    def __call__(self, obj, t, args={}):
        return -obj


class _Block_Sum_Qo(_Block_transform):
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
        return obj + self.other(t, args)


class _Block_mul(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj * self.other(t, args)


class _Block_rmul(_Block_transform):
    def __call__(self, obj, t, args={}):
        return self.other(t, args) * obj


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
        return obj.dag()


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
    def __call__(self, obj, t, args={}):
        c_ops = [op(t, args) for op in self.other]
        return liouvillian(obj, c_ops)


class _Block_lindblad_dissipator(_Block_transform):
    def __call__(self, obj, t, args={}):
        return lindblad_dissipator(obj, chi=self.other)


class _Block_apply(_Block_transform):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def copy(self):
        return _Block_apply(self.func, args, kwargs.copy())

    def __call__(self, obj, t, args={}):
        return self.func(obj, *self.args, **self.kwargs)

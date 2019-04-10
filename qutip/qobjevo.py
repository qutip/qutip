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
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, spmv, cy_spmm_tr)
from qutip.cy.cqobjevo import (CQobjCte, CQobjCteDense, CQobjEvoTd,
                                 CQobjEvoTdMatched, CQobjEvoTdDense)
from qutip.cy.cqobjevo_factor import (InterCoeffT, InterCoeffCte,
                                      InterpolateCoeff, StrCoeff)
import pickle
import sys
import scipy
import os

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
    "zerf": scipy.special.erf,
    "sqrt": np.sqrt,
    "real": np.real,
    "imag": np.imag,
    "conj": np.conj,
    "abs": np.abs,
    "norm": lambda x: np.abs(x)**2,
    "arg": np.angle,
    "proj": proj,
    "np": np,
    "spe": scipy.special}

class _file_list:
    """
    Contain temp a list .pyx to clean
    """
    def __init__(self):
        self.files = []

    def add(self, file_):
        self.files += [file_ + ".pyx"]

    def clean(self):
        for i, file_ in enumerate(self.files):
            try:
                os.remove(file_)
            except:
                pass
            if not os.path.isfile(file_):
                # Don't exist anymore
                del self.files[i]

    def __del__(self):
        self.clean()

coeff_files = _file_list()


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


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, coeff_func):
        self.coeff_func = coeff_func

    def __call__(self, t, args={}):
        return self.coeff_func(t, args["_state_vec"], args)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# object for each time dependent element of the QobjEvo
# qobj : the Qobj of element ([*Qobj*, f])
# get_coeff : a callable that take (t, args) and return the coeff at that t
# coeff : The coeff as a string, array or function as provided by the user.
# type : flag for the type of coeff
class EvoElement:
    def __init__(self, qobj, get_coeff, coeff, type):
        self.qobj = qobj
        self.get_coeff = get_coeff
        self.coeff = coeff
        self.type = type

    @classmethod
    def make(cls, list_):
        """self.qobj = list_[0]
        self.get_coeff = list_[1]
        self.coeff = list_[2]
        self.type = list_[3]"""
        return cls(*list_)

    def __getitem__(self, i):
        if i == 0:
            return self.qobj
        if i == 1:
            return self.get_coeff
        if i == 2:
            return self.coeff
        if i == 3:
            return self.type


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

    def __init__(self, Q_object=[], args={}, tlist=None, copy=True):
        if isinstance(Q_object, QobjEvo):
            if copy:
                self._inplace_copy(Q_object)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args)
            return

        self.const = False
        self.dummy_cte = False
        self.args = args.copy() if copy else args
        self.dynamics_args = []
        self.cte = None
        self.tlist = tlist
        self.compiled = ""
        self.compiled_qobjevo = None
        self.compiled_ptr = None
        self.coeff_get = None
        self.type = "none"
        self.omp = 0

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
                self.type = "cte"
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
                    self.ops.append(EvoElement(op[0], op[1], op[1], "func"))
                elif type_ == 2:
                    op_type_count[1] += 1
                    self.ops.append(EvoElement(op[0], _StrWrapper(op[1]),
                                    op[1], "string"))
                elif type_ == 3:
                    op_type_count[2] += 1
                    self.ops.append(EvoElement(op[0], _CubicSplineWrapper(tlist, op[1]),
                                     op[1].copy(), "array"))
                elif type_ == 4:
                    op_type_count[3] += 1
                    self.ops.append(EvoElement(op[0], op[1], op[1], "spline"))

            nops = sum(op_type_count)
            if op_type_count[0] == nops:
                self.type = "func"
            elif op_type_count[1] == nops:
                self.type = "string"
            elif op_type_count[2] == nops:
                self.type = "array"
            elif op_type_count[3] == nops:
                self.type = "spline"
            elif op_type_count[0]:
                self.type = "mixed_callable"
            else:
                self.type = "mixed_compilable"

            try:
                if not self.cte:
                    self.cte = self.ops[0].qobj
                    # test is all qobj are compatible (shape, dims)
                    for op in self.ops[1:]:
                        self.cte += op.qobj
                    self.cte *= 0.
                    self.dummy_cte = True
                else:
                    cte_copy = self.cte.copy()
                    # test is all qobj are compatible (shape, dims)
                    for op in self.ops:
                        cte_copy += op.qobj
            except Exception as e:
                raise TypeError("Qobj not compatible.") from e

            if not self.ops:
                self.const = True
        self.num_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)
        self._args_checks()

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

    def _check_old_with_state(self):
        add_vec = False
        for op in self.ops:
            if op.type == "func":
                try:
                    op.get_coeff(0., self.args)
                except TypeError as e:
                    nfunc = _StateAsArgs(self.coeff)
                    op = EvoElement((op.qobj, nfunc, nfunc, "func"))
                    add_vec = True
        if add_vec:
            self.dynamics_args += [("_state_vec", "vec", None)]

    def __del__(self):
        # sometime not called
        coeff_files.clean()

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
            old_compiled = self.compiled
            self.compiled = False
            self.args.update(args)
            op_t = self.__call__(t, data=data)
            self.args = old_args
            self.compiled = old_compiled
        elif self.const:
            if data:
                op_t = self.cte.data.copy()
            else:
                op_t = self.cte.copy()
        elif self.compiled and self.compiled.split()[0] != "dense":
            op_t = self.compiled_qobjevo.call(t, data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part.qobj.data * part.get_coeff(t, self.args)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part.qobj * part.get_coeff(t, self.args)

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

    def copy(self):
        new = QobjEvo(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.dynamics_args = self.dynamics_args.copy()
        new.tlist = self.tlist
        new.dummy_cte = self.dummy_cte
        new.num_obj = self.num_obj
        new.type = self.type
        new.compiled = False
        new.compiled_qobjevo = None
        new.compiled_ptr = None
        new.coeff_get = None
        new.coeff_files = []

        for op in self.ops:
            if op.type == "array":
                new_coeff = op.coeff.copy()
            else:
                new_coeff = op.coeff
            new.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                      new_coeff, op.type))

        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.dynamics_args = other.dynamics_args
        self.tlist = other.tlist
        self.dummy_cte = other.dummy_cte
        self.num_obj = other.num_obj
        self.type = other.type
        self.compiled = ""
        self.compiled_qobjevo = None
        self.compiled_ptr = None
        self.coeff_get = None
        self.ops = []

        for op in other.ops:
            if op.type == "array":
                new_coeff = op.coeff.copy()
            else:
                new_coeff = op.coeff
            self.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                       new_coeff, op.type))

    def arguments(self, args):
        if not isinstance(args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(args)
        self._args_checks(True)
        if self.compiled and self.compiled.split()[2] is not "cte":
            if isinstance(self.coeff_get, StrCoeff):
                self.coeff_get.set_args(self.args)
                self.coeff_get._set_dyn_args(self.dynamics_args)
            elif isinstance(self.coeff_get, _UnitedFuncCaller):
                self.coeff_get.set_args(self.args, self.dynamics_args)
            else:
                pass

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
                if op.type == "array":
                    new_coeff = op.coeff.copy()
                else:
                    new_coeff = op.coeff
                self.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                           new_coeff, op.type))
                l += 1
            self.args.update(**other.args)
            self.dynamics_args += other.dynamics_args
            self.const = self.const and other.const
            self.dummy_cte = self.dummy_cte and other.dummy_cte
            if self.type != other.type:
                if self.type in ["func", "mixed_callable"] or \
                        other.type in ["func", "mixed_callable"]:
                    self.type = "mixed_callable"
                else:
                    self.type = "mixed_compilable"
            self.compiled = ""
            self.compiled_qobjevo = None
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

        self.num_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)
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
                op.qobj = other * op.qobj
            return res
        else:
            res *= other
            return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, Number):
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
                self.dynamics_args += other.dynamics_args
                self.dummy_cte = self.dummy_cte and other.dummy_cte
                self.num_obj = (len(self.ops) if
                              self.dummy_cte else len(self.ops) + 1)
            self._reset_type()

        else:
            raise TypeError("QobjEvo can only be multiplied"
                            " with QobjEvo, Qobj or numbers")
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
        res.cte = -res.cte
        for op in res.ops:
            op.qobj = -op.qobj
        return res

    def _ops_mul_(self, opL, opR):
        new_f = _Prod(opL.get_coeff, opR.get_coeff)
        new_op = [opL.qobj*opR.qobj, new_f, None, 0]
        if opL.type == opR.type and opL.type == "string":
            new_op[2] = "(" + opL.coeff + ") * (" + opR.coeff + ")"
            new_op[3] = "string"
        elif opL[3] == opR[3] and opL[3] == "array":
            new_op[2] = opL[2]*opR[2]
            new_op[3] = "array"
        else:
            new_op[2] = new_f
            new_op[3] = "func"
            if self.type not in ["func", "mixed_callable"]:
                self.type = "mixed_callable"
        return EvoElement.make(new_op)

    def _ops_mul_cte(self, op, cte, side):
        new_op = [None, op.get_coeff, op.coeff, op.type]
        if side == "R":
            new_op[0] = op.qobj * cte
        if side == "L":
            new_op[0] = cte * op.qobj
        return EvoElement.make(new_op)

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
        for op in res.ops:
            op.qobj = op.qobj.conj()
        res._f_conj()
        return res

    def dag(self):
        res = self.copy()
        res.cte = res.cte.dag()
        for op in res.ops:
            op.qobj = op.qobj.dag()
        res._f_conj()
        return res

    def _cdc(self):
        """return a.dag * a """
        if not self.num_obj == 1:
            res = self.dag()
            res *= self
        else:
            res = self.copy()
            res.cte = res.cte.dag() * res.cte
            for op in res.ops:
                op.qobj = op.qobj.dag() * op.qobj
            res._f_norm2()
        return res

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
            already_matched = False
            for _set in sets:
                already_matched = already_matched or i in _set
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.qobj == op2.qobj:
                        same_flag = op1.type == op2.type
                        callable_1 = op1.type in callable_flags
                        callable_2 = op2.type in callable_flags
                        if (same_flag or (callable_1 and callable_2)):
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
                    if op1.type != op2.type:
                        pass
                    elif op1.type is "array":
                        if np.allclose(op1.coeff, op2.coeff):
                            this_set.append(j+i+1)
                    else:
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

            elif self.ops[_set[0]].type in callable_flags:
                new_op = [self.ops[_set[0]].qobj, None, None, "func"]
                fs = []
                for i in _set:
                    fs += [self.ops[i].get_coeff]
                new_op[1] = _Add(fs)
                new_op[2] = new_op[1]
                new_ops.append(EvoElement.make(new_op))

            elif self.ops[_set[0]].type is "string":
                new_op = [self.ops[_set[0]].qobj, None, None, "string"]
                new_str = "(" + self.ops[_set[0]].coeff + ")"
                for i in _set[1:]:
                    new_str += " + (" + self.ops[i].coeff + ")"
                new_op[1] = _StrWrapper(new_str)
                new_op[2] = new_str
                new_ops.append(EvoElement.make(new_op))

            elif self.ops[_set[0]].type is "array":
                new_op = [self.ops[_set[0]].qobj, None, None, "array"]
                new_array = (self.ops[_set[0]].coeff).copy()
                for i in _set[1:]:
                    new_array += self.ops[i].coeff
                new_op[2] = new_array
                new_op[1] = _CubicSplineWrapper(self.tlist, new_array)
                new_ops.append(EvoElement.make(new_op))

        self.ops = new_ops

    def _compress_merge_func(self, fsets):
        new_ops = []
        for _set in fsets:
            base = self.ops[_set[0]]
            new_op = [None, base.get_coeff, base.coeff, base.type]
            if len(_set) == 1:
                new_op[0] = base.qobj
            else:
                new_op[0] = base.qobj.copy()
                for i in _set[1:]:
                    new_op[0] += self.ops[i].qobj
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops

    def compress(self):
        self.tidyup()
        sets, fsets = self._compress_make_set()
        N_sets = len(sets)
        N_fsets = len(fsets)
        num_ops = len(self.ops)

        if N_sets < num_ops and N_fsets < num_ops:
            # Both could be better
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
            if N_sets < N_fsets:
                self._compress_merge_qobj(sets)
            else:
                self._compress_merge_func(fsets)
            sets, fsets = self._compress_make_set()
            N_sets = len(sets)
            N_fsets = len(fsets)
            num_ops = len(self.ops)

        if N_sets < num_ops:
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
            self._compress_merge_qobj(sets)
        elif N_fsets < num_ops:
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
            self._compress_merge_func(fsets)
        self._reset_type()

    def _reset_type(self):
        op_type_count = [0, 0, 0, 0]
        for op in self.ops:
            if op.type == "func":
                op_type_count[0] += 1
            elif op.type == "string":
                op_type_count[1] += 1
            elif op.type == "array":
                op_type_count[2] += 1
            elif op.type == "spline":
                op_type_count[3] += 1

        nops = sum(op_type_count)
        if op_type_count[0] == nops:
            self.type = "func"
        elif op_type_count[1] == nops:
            self.type = "string"
        elif op_type_count[2] == nops:
            self.type = "array"
        elif op_type_count[3] == nops:
            self.type = "spline"
        elif op_type_count[0]:
            self.type = "mixed_callable"
        else:
            self.type = "mixed_compilable"

        self.num_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)

    def permute(self, order):
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op.qobj = op.qobj.permute(order)
        return res

    def ptrace(self, sel):
        res = self.copy()
        res.cte = res.cte.ptrace(sel)
        for op in res.ops:
            op.qobj = op.qobj.ptrace(sel)
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
            op.get_coeff = function(op.get_coeff, *args, **kw_args)
            if op.type == ["func", "spline"]:
                op.coeff = op.get_coeff
                op.type = "func"
            elif op.type == "string":
                if str_mod is None:
                    op.coeff = op.get_coeff
                    op.type = "func"
                else:
                    op.coeff = str_mod[0] + op.coeff + str_mod[1]
            elif op.type == "array":
                if inplace_np:
                    # keep the original function, change the array
                    def f(a):
                        return a
                    ff = function(f, *args, **kw_args)
                    for i, v in enumerate(op.coeff):
                        op.coeff[i] = ff(v)
                    op.get_coeff = _CubicSplineWrapper(self.tlist, op.coeff)
                else:
                    op.coeff = op.get_coeff
                    op.type = "func"
        if self.type == "string" and str_mod is None:
            res.type = "mixed_callable"
        elif self.type == "array" and not inplace_np:
            res.type = "mixed_callable"
        elif self.type == "spline":
            res.type = "mixed_callable"
        elif self.type == "mixed_compilable":
            for op in res.ops:
                if op.type == "func":
                    res.type = "mixed_callable"
        return res

    def _f_norm2(self):
        new_ops = []
        for op in self.ops:
            new_op = [op.qobj, None, None, op.type]
            if op.type == "func":
                new_op[1] = _Norm2(op.get_coeff)
                new_op[2] = new_op[1]
            elif op.type == "string":
                new_op[2] = "norm(" + op.coeff + ")"
                new_op[1] = _StrWrapper(new_op[2])
            elif op.type == "array":
                new_op[2] = np.abs(op.coeff)**2
                new_op[1] = _CubicSplineWrapper(self.tlist, new_op[2])
            elif op.type == "spline":
                new_op[1] = _Norm2(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops
        return self

    def _f_conj(self):
        new_ops = []
        for op in self.ops:
            new_op = [op.qobj, None, None, op.type]
            if op.type == "func":
                new_op[1] = _Conj(op.get_coeff)
                new_op[2] = new_op[1]
            elif op.type == "string":
                new_op[2] = "conj(" + op.coeff + ")"
                new_op[1] = _StrWrapper(new_op[2])
            elif op.type == "array":
                new_op[2] = np.conj(op.coeff)
                new_op[1] = _CubicSplineWrapper(self.tlist, new_op[2])
            elif op.type == "spline":
                new_op[1] = _Conj(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops
        return self

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
            if self.compiled:
                exp = self.compiled_qobjevo.expect(t, vec)
            elif self.cte.issuper:
                self._dynamics_args_update(t, state)
                exp = cy_expect_rho_vec(self.__call__(t, data=True), vec, 0)
            else:
                self._dynamics_args_update(t, state)
                exp = cy_expect_psi(self.__call__(t, data=True), vec, 0)
        elif vec.shape[0] == self.cte.shape[1]**2:
            if self.compiled:
                exp = self.compiled_qobjevo.overlapse(t, vec)
            else:
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

        if self.compiled:
            out = self.compiled_qobjevo.mul_vec(t, vec)
        else:
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

        if self.compiled:
            out = self.compiled_qobjevo.mul_mat(t, mat)
        else:
            self._dynamics_args_update(t, mat)
            out = self.__call__(t, data=True) * mat

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def compile(self, code=False, matched=False, dense=False, omp=0):
        self.tidyup()
        Code = None
        if self.compiled:
            return
        for _, _, op in self.dynamics_args:
            if isinstance(op, QobjEvo):
                op.compile(code, matched, dense, omp)
        if not qset.has_openmp:
            omp = 0
        if omp:
            nnz = [self.cte.data.nnz]
            for part in self.ops:
                nnz += [part.qobj.data.nnz]
            if all(qset.openmp_thresh < nz for nz in nnz):
                omp = 0
        if self.const:
            if dense:
                self.compiled_qobjevo = CQobjCteDense()
                self.compiled = "dense single cte"
            elif omp:
                self.compiled_qobjevo = CQobjCteOmp()
                self.compiled = "csr omp cte"
                self.compiled_qobjevo.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_qobjevo = CQobjCte()
                self.compiled = "csr single cte"
            self.compiled_qobjevo.set_data(self.cte)
        else:
            if matched:
                if omp:
                    self.compiled_qobjevo = CQobjEvoTdMatchedOmp()
                    self.compiled = "matched omp "
                    self.compiled_qobjevo.set_threads(omp)
                    self.omp = omp
                else:
                    self.compiled_qobjevo = CQobjEvoTdMatched()
                    self.compiled = "matched single "
            elif dense:
                self.compiled_qobjevo = CQobjEvoTdDense()
                self.compiled = "dense single "
            elif omp:
                self.compiled_qobjevo = CQobjEvoTdOmp()
                self.compiled = "csr omp "
                self.compiled_qobjevo.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_qobjevo = CQobjEvoTd()
                self.compiled = "csr single "
            self.compiled_qobjevo.set_data(self.cte, self.ops)
            self.compiled_qobjevo.has_dyn_args(bool(self.dynamics_args))

            if self.type in ["func"]:
                funclist = []
                for part in self.ops:
                    funclist.append(part.get_coeff)
                self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                                   self.dynamics_args, self.cte)
                self.compiled += "pyfunc"
                self.compiled_qobjevo.set_factor(func=self.coeff_get)
            elif self.type in ["mixed_callable"]:
                funclist = []
                for part in self.ops:
                    if isinstance(part.get_coeff, _StrWrapper):
                        part.get_coeff, file = _compile_str_single(part.coeff, self.args)
                        coeff_files.add(file)
                    funclist.append(part.get_coeff)
                self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                                   self.dynamics_args, self.cte)
                self.compiled += "pyfunc"
                self.compiled_qobjevo.set_factor(func=self.coeff_get)
            elif self.type in ["string", "mixed_compilable"]:
                # All factor can be compiled
                self.coeff_get, Code, file = _compiled_coeffs(self.ops,
                                                              self.args,
                                                              self.dynamics_args,
                                                              self.tlist)
                coeff_files.add(file)
                self.compiled_qobjevo.set_factor(obj=self.coeff_get)
                self.compiled += "cyfactor"
            elif self.type == "array":
                if np.allclose(np.diff(self.tlist),
                               self.tlist[1] - self.tlist[0]):
                    self.coeff_get = InterCoeffCte(self.ops, None,
                                                     self.tlist)
                else:
                    self.coeff_get = InterCoeffT(self.ops, None, self.tlist)
                self.compiled += "cyfactor"
                self.compiled_qobjevo.set_factor(obj=self.coeff_get)
            elif self.type == "spline":
                self.coeff_get = InterpolateCoeff(self.ops, None, None)
                self.compiled += "cyfactor"
                self.compiled_qobjevo.set_factor(obj=self.coeff_get)
            else:
                pass
            coeff_files.clean()
            if code:
                return Code

    def _get_coeff(self, t):
        out = []
        for part in self.ops:
            out.append(part.get_coeff(t, self.args))
        return out

    def __getstate__(self):
        _dict_ = {key: self.__dict__[key]
                  for key in self.__dict__ if key is not "compiled_qobjevo"}
        if self.compiled:
            return (_dict_, self.compiled_qobjevo.__getstate__())
        else:
            return (_dict_,)

    def __setstate__(self, state):
        self.__dict__ = state[0]
        self.compiled_qobjevo = None
        if self.compiled:
            mat_type, threading, td =  self.compiled.split()
            if mat_type == "csr":
                if safePickle:
                    # __getstate__ and __setstate__ of compiled_qobjevo pass pointers
                    # In 'safe' mod, these pointers are not used.
                    if td == "cte":
                        if threading == "single":
                            self.compiled_qobjevo = CQobjCte()
                            self.compiled_qobjevo.set_data(self.cte)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjCteOmp()
                            self.compiled_qobjevo.set_data(self.cte)
                            self.compiled_qobjevo.set_threads(self.omp)
                    else:
                        # time dependence is pyfunc or cyfactor
                        if threading == "single":
                            self.compiled_qobjevo = CQobjEvoTd()
                            self.compiled_qobjevo.set_data(self.cte, self.ops)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjEvoTdOmp()
                            self.compiled_qobjevo.set_data(self.cte, self.ops)
                            self.compiled_qobjevo.set_threads(self.omp)

                        if td == "pyfunc":
                            self.compiled_qobjevo.set_factor(obj=self.coeff_get)
                        elif td == "cyfactor":
                            self.compiled_qobjevo.set_factor(func=self.coeff_get)
                else:
                    if td == "cte":
                        if threading == "single":
                            self.compiled_qobjevo = CQobjCte.__new__(CQobjCte)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjCteOmp.__new__(CQobjCteOmp)
                            self.compiled_qobjevo.set_threads(self.omp)
                    else:
                        # time dependence is pyfunc or cyfactor
                        if threading == "single":
                            self.compiled_qobjevo = CQobjEvoTd.__new__(CQobjEvoTd)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjEvoTdOmp.__new__(CQobjEvoTdOmp)
                            self.compiled_qobjevo.set_threads(self.omp)
                    self.compiled_qobjevo.__setstate__(state[1])

            elif mat_type == "dense":
                if td == "cte":
                    self.compiled_qobjevo = \
                        CQobjCteDense.__new__(CQobjCteDense)
                else:
                    CQobjEvoTdDense.__new__(CQobjEvoTdDense)
                self.compiled_qobjevo.__setstate__(state[1])

            elif mat_type == "matched":
                if threading == "single":
                    self.compiled_qobjevo = \
                        CQobjEvoTdMatched.__new__(CQobjEvoTdMatched)
                elif threading == "omp":
                    self.compiled_qobjevo = \
                        CQobjEvoTdMatchedOmp.__new__(CQobjEvoTdMatchedOmp)
                    self.compiled_qobjevo.set_threads(self.omp)
                self.compiled_qobjevo.__setstate__(state[1])


# Function defined inside another function cannot be pickled,
# Using class instead
class _UnitedFuncCaller:
    def __init__(self, funclist, args, dynamics_args, cte):
        self.funclist = funclist
        self.args = args
        self.dynamics_args = dynamics_args
        self.dims = cte.dims
        self.shape = cte.shape

    def set_args(self, args, dynamics_args):
        self.args = args
        self.dynamics_args = dynamics_args

    def dyn_args(self, t, state, shape):
        # 1d array are to F ordered
        mat = state.reshape(shape, order="F")
        for name, what, op in self.dynamics_args:
            if what == "vec":
                self.args[name] = state
            elif what == "mat":
                self.args[name] = mat
            elif what == "Qobj":
                if self.shape[1] == shape[1]:  # oper
                    self.args[name] = Qobj(mat, dims=self.dims)
                elif shape[1] == 1:
                    self.args[name] = Qobj(mat, dims=[self.dims[1],[1]])
                else:  # rho
                    self.args[name] = Qobj(mat, dims=self.dims[1])
            elif what == "expect":  # ket
                if shape[1] == op.cte.shape[1]: # same shape as object
                    self.args[name] = op.mul_mat(t, mat).trace()
                else:
                    self.args[name] = op.expect(t, state)

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

    def get_args(self):
        return self.args


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

from qutip.superoperator import vec2mat

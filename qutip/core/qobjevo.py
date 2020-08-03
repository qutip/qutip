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

from .interpolate import Cubic_Spline
from .qobj import Qobj
from .qobjevo_codegen import (
    _compile_str_single, _compiled_coeffs, _compiled_coeffs_python,
)
from .superoperator import stack_columns, unstack_columns
from .cy.spmatfuncs import (
    cy_expect_rho_vec, cy_expect_psi, spmv,
)
from .cy.cqobjevo import CQobjEvo
from .cy.cqobjevo_factor import (
    InterCoeffT, InterCoeffCte, InterpolateCoeff, StrCoeff, StepCoeffCte,
    StepCoeffT,
)

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


def proj(x):
    return x if np.isfinite(x) else (np.inf + 0j*np.imag(x))


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
    "spe": scipy.special,
}


class _file_list:
    """
    Contain temp a list .pyx to clean
    """
    def __init__(self):
        self.files = []

    def add(self, file_):
        self.files += [file_ + ".pyx"]

    def clean(self):
        to_del = []
        for i, file_ in enumerate(self.files):
            try:
                os.remove(file_)
                to_del.append(i)
            except Exception:
                if not os.path.isfile(file_):
                    to_del.append(i)

        for i in to_del[::-1]:
            del self.files[i]

    def __del__(self):
        self.clean()


coeff_files = _file_list()


class _StrWrapper:
    def __init__(self, code):
        self.code = "_out = " + code

    def __call__(self, t, args=None):
        env = {"t": t}
        if args is not None:
            env.update(args)
        exec(self.code, str_env, env)
        return env["_out"]


class _CubicSplineWrapper:
    # Using scipy's CubicSpline since Qutip's one
    # only accept linearly distributed tlist
    def __init__(self, tlist, coeff, args=None):
        self.coeff = coeff
        self.tlist = tlist
        if args.get("_step_func_coeff", False):
            self.func = interp1d(
                self.tlist, self.coeff, kind="previous",
                bounds_error=False, fill_value=0.)
        else:
            self.func = CubicSpline(self.tlist, self.coeff)

    def __call__(self, t, args=None):
        return self.func([t])[0]


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, coeff_func):
        self.coeff_func = coeff_func

    def __call__(self, t, args):
        return self.coeff_func(t, args["_state_vec"], args)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class StateArgs:
    """Object to indicate to use the state in args outside solver.
    args[key] = StateArgs(type, op)
    """
    def __init__(self, type="Qobj", op=None):
        self.dyn_args = (type, op)

    def __call__(self):
        return self.dyn_args


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
        self.coeff_get = None
        self.type = "none"
        self.omp = 0
        self.coeff_files = []
        self.use_cython = use_cython[0]

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
                    self.ops.append(EvoElement(
                        op[0],
                        _CubicSplineWrapper(tlist, op[1], args=self.args),
                        op[1].copy(), "array"))
                elif type_ == 4:
                    op_type_count[3] += 1
                    self.ops.append(EvoElement(op[0], op[1], op[1], "spline"))

            nops = sum(op_type_count)
            if all([op_t == 0 for op_t in op_type]):
                self.type = "cte"
            elif op_type_count[0] == nops:
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
        if e_ops:
            for i, dargs in enumerate(self.dynamics_args):
                if dargs[1] == "expect" and isinstance(dargs[2], int):
                    self.dynamics_args[i] = (dargs[0], "expect",
                                             QobjEvo(e_ops[dargs[2]]))
        if state0 is not None:
            self._dynamics_args_update(0., state0)

    def _td_format_check_single(self, Q_object, tlist=None):
        op_type = []

        if isinstance(Q_object, Qobj):
            op_type = 0
        elif isinstance(Q_object, (FunctionType,
                                   BuiltinFunctionType, functools.partial)):
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

            if isinstance(self.args[key], StateArgs):
                self.dynamics_args += [(key, *self.args[key]())]
                self.args[key] = 0.

    def _check_old_with_state(self):
        add_vec = False
        for op in self.ops:
            if op.type == "func":
                try:
                    op.get_coeff(0., self.args)
                except TypeError:
                    nfunc = _StateAsArgs(self.coeff)
                    op = EvoElement(op.qobj, nfunc, nfunc, "func")
                    add_vec = True
        if add_vec:
            self.dynamics_args += [("_state_vec", "vec", None)]

    def __del__(self):
        for file_ in self.coeff_files:
            try:
                os.remove(file_)
            except:
                pass

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
            op_t = self.compiled_qobjevo.call(t, data=data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part.qobj.data * part.get_coeff(t, self.args)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part.qobj * part.get_coeff(t, self.args)

        return op_t

    def _dynamics_args_update(self, t, state: _data.Data):
        for name, what, e_op in self.dynamics_args:
            self.args[name] = _dynamic_argument(t, self.cte, state, what, e_op)

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
        new.coeff_get = None
        new.coeff_files = []
        new.use_cython = self.use_cython

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
        self.coeff_get = None
        self.ops = []
        self.coeff_files = []
        self.use_cython = other.use_cython

        for op in other.ops:
            if op.type == "array":
                new_coeff = op.coeff.copy()
            else:
                new_coeff = op.coeff
            self.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                       new_coeff, op.type))

    def arguments(self, new_args):
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        # remove dynamics_args that are to be refreshed
        self.dynamics_args = [dargs for dargs in self.dynamics_args
                              if dargs[0] not in new_args]
        self.args.update(new_args)
        self._args_checks()
        if self.compiled and self.compiled.split()[2] != "cte":
            if isinstance(self.coeff_get, StrCoeff):
                self.coeff_get.set_args(self.args)
                self.coeff_get._set_dyn_args(self.dynamics_args)
            elif isinstance(self.coeff_get, _UnitedFuncCaller):
                self.coeff_get.set_args(self.args, self.dynamics_args)

    def solver_set_args(self, new_args, state: _data.Data, e_ops):
        if not isinstance(state, _data.Data):
            raise TypeError("state should be a data-layer object")
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

    def _ops_mul_(self, opL, opR):
        new_f = _Prod(opL.get_coeff, opR.get_coeff)
        new_op = EvoElement(opL.qobj*opR.qobj, new_f, None, 0)
        if opL.type == opR.type and opL.type == "string":
            new_op.coeff = "(" + opL.coeff + ") * (" + opR.coeff + ")"
            new_op.type = "string"
        elif opL.type == opR.type and opL.type == "array":
            new_op.coeff = opL.coeff * opR.coeff
            new_op.type = "array"
        else:
            new_op.coeff = new_f
            new_op.type = "func"
            if self.type not in ["func", "mixed_callable"]:
                self.type = "mixed_callable"
        return new_op

    def _ops_mul_cte(self, op, cte, side):
        new_op = EvoElement(None, op.get_coeff, op.coeff, op.type)
        if side == "R":
            new_op.qobj = op.qobj * cte
        if side == "L":
            new_op.qobj = cte * op.qobj
        return new_op

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
                    elif op1.type == "array":
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
                new_ops.append(EvoElement(*new_op))

            elif self.ops[_set[0]].type == "string":
                new_op = [self.ops[_set[0]].qobj, None, None, "string"]
                new_str = "(" + self.ops[_set[0]].coeff + ")"
                for i in _set[1:]:
                    new_str += " + (" + self.ops[i].coeff + ")"
                new_op[1] = _StrWrapper(new_str)
                new_op[2] = new_str
                new_ops.append(EvoElement(*new_op))

            elif self.ops[_set[0]].type == "array":
                new_op = [self.ops[_set[0]].qobj, None, None, "array"]
                new_array = (self.ops[_set[0]].coeff).copy()
                for i in _set[1:]:
                    new_array += self.ops[i].coeff
                new_op[2] = new_array
                new_op[1] = _CubicSplineWrapper(
                    self.tlist, new_array, args=self.args)
                new_ops.append(EvoElement(*new_op))

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
            new_ops.append(EvoElement(*new_op))
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
        if not self.ops and self.dummy_cte is False:
            self.type = "cte"
        elif op_type_count[0] == nops:
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

    def apply_decorator(self, function, *args,
                        str_mod=None, inplace_np=None, **kwargs):
        res = self.copy()
        for op in res.ops:
            op.get_coeff = function(op.get_coeff, *args, **kwargs)
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
                    ff = function(f, *args, **kwargs)
                    for i, v in enumerate(op.coeff):
                        op.coeff[i] = ff(v)
                    op.get_coeff = _CubicSplineWrapper(
                        self.tlist, op.coeff, args=self.args)
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
        self.compiled = ""
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
                new_op[1] = _CubicSplineWrapper(
                    self.tlist, new_op[2], args=self.args)
            elif op.type == "spline":
                new_op[1] = _Norm2(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement(*new_op))
        self.ops = new_ops
        return self

    def _f_conj(self):
        self.compiled = ""
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
                new_op[1] = _CubicSplineWrapper(
                    self.tlist, new_op[2], args=self.args)
            elif op.type == "spline":
                new_op[1] = _Conj(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement(*new_op))
        self.ops = new_ops
        return self

    def _shift(self):
        self.compiled = ""
        self.args.update({"_t0": 0})
        new_ops = []
        for op in self.ops:
            new_op = [op.qobj, None, None, op.type]
            if op.type == "func":
                new_op[1] = _Shift(op.get_coeff)
                new_op[2] = new_op[1]
            elif op.type == "string":
                new_op[2] = re.sub(r"(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                                   "(t+_t0)", " " + op.coeff + " ")
                new_op[1] = _StrWrapper(new_op[2])
            elif op.type == "array":
                new_op[2] = _Shift(op.get_coeff)
                new_op[1] = new_op[2]
                new_op[3] = "func"
                self.type = "mixed_callable"
            elif op.type == "spline":
                new_op[1] = _Shift(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement(*new_op))
        self.ops = new_ops
        return self

    def expect(self, t, state, herm=0):
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

        if self.compiled:
            exp = self.compiled_qobjevo.expect(t, state)
        elif self.cte.issuper:
            state = _data.column_stack_dense(state)
            self._dynamics_args_update(t, state)
            exp = _data.expect_super_csr_dense(self.__call__(t, data=True),
                                               state)
        else:
            self._dynamics_args_update(t, state)
            exp = _data.expect_csr_dense(self.__call__(t, data=True), state)
        return exp.real if herm else exp

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
            out = _data.matmul_csr_dense_dense(self.__call__(t, data=True),
                                               vec).as_ndarray()[:, 0]

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
            nnz = [_data.csr.nnz(self.cte.data)]
            for part in self.ops:
                nnz += [_data.csr.nnz(part.qobj.data)]
            if all(qset.openmp_thresh < nz for nz in nnz):
                omp = 0

        if self.const:
            if dense:
                self.compiled_qobjevo = CQobjEvo()
                self.compiled = "dense single cte"
            elif omp:
                self.compiled_qobjevo = CQobjCteOmp()
                self.compiled = "csr omp cte"
                self.compiled_qobjevo.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_qobjevo = CQobjEvo()
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
                    self.compiled_qobjevo = CQobjEvo()
                    self.compiled = "matched single "
            elif dense:
                self.compiled_qobjevo = CQobjEvo()
                self.compiled = "dense single "
            elif omp:
                self.compiled_qobjevo = CQobjEvoTdOmp()
                self.compiled = "csr omp "
                self.compiled_qobjevo.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_qobjevo = CQobjEvo()
                self.compiled = "csr single "
            self.compiled_qobjevo.set_data(self.cte, self.ops)
            self.compiled_qobjevo.has_dyn_args(bool(self.dynamics_args))

            if self.type in ["func"]:
                # funclist = []
                # for part in self.ops:
                #    funclist.append(part.get_coeff)
                funclist = [part.get_coeff for part in self.ops]
                self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                                   self.dynamics_args, self.cte)
                self.compiled += "pyfunc"
                self.compiled_qobjevo.set_factor(func=self.coeff_get)

            elif self.type in ["mixed_callable"] and self.use_cython:
                funclist = []
                for part in self.ops:
                    if isinstance(part.get_coeff, _StrWrapper):
                        get_coeff, file_ = _compile_str_single(
                                                                part.coeff,
                                                                self.args)
                        coeff_files.add(file_)
                        self.coeff_files.append(file_)
                        funclist.append(get_coeff)
                    else:
                        funclist.append(part.get_coeff)

                self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                                   self.dynamics_args,
                                                   self.cte)
                self.compiled += "pyfunc"
                self.compiled_qobjevo.set_factor(func=self.coeff_get)
            elif self.type in ["mixed_callable"]:
                funclist = [part.get_coeff for part in self.ops]
                _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(
                                                        self.ops,
                                                        self.args,
                                                        self.dynamics_args,
                                                        self.tlist)
                coeff_files.add(file_)
                self.coeff_files.append(file_)
                self.coeff_get = _UnitedStrCaller(funclist, self.args,
                                                  self.dynamics_args,
                                                  self.cte)
                self.compiled_qobjevo.set_factor(func=self.coeff_get)
                self.compiled += "pyfunc"
            elif self.type in ["string", "mixed_compilable"]:
                if self.use_cython:
                    # All factor can be compiled
                    self.coeff_get, Code, file_ = _compiled_coeffs(
                                                        self.ops,
                                                        self.args,
                                                        self.dynamics_args,
                                                        self.tlist)
                    coeff_files.add(file_)
                    self.coeff_files.append(file_)
                    self.compiled_qobjevo.set_factor(obj=self.coeff_get)
                    self.compiled += "cyfactor"
                else:
                    # All factor can be compiled
                    _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(
                                                        self.ops,
                                                        self.args,
                                                        self.dynamics_args,
                                                        self.tlist)
                    coeff_files.add(file_)
                    self.coeff_files.append(file_)
                    funclist = [part.get_coeff for part in self.ops]
                    self.coeff_get = _UnitedStrCaller(funclist, self.args,
                                                      self.dynamics_args,
                                                      self.cte)
                    self.compiled_qobjevo.set_factor(func=self.coeff_get)
                    self.compiled += "pyfunc"

            elif self.type == "array":
                try:
                    use_step_func = self.args["_step_func_coeff"]
                except KeyError:
                    use_step_func = 0
                if np.allclose(np.diff(self.tlist),
                            self.tlist[1] - self.tlist[0]):
                    if use_step_func:
                        self.coeff_get = StepCoeffCte(
                            self.ops, None, self.tlist)
                    else:
                        self.coeff_get = InterCoeffCte(
                            self.ops, None, self.tlist)
                else:
                    if use_step_func:
                        self.coeff_get = StepCoeffT(
                            self.ops, None, self.tlist)
                    else:
                        self.coeff_get = InterCoeffT(
                            self.ops, None, self.tlist)
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
        return [part.get_coeff(t, self.args) for part in self.ops]

    def __getstate__(self):
        _dict_ = self.__dict__.copy()
        # TODO: get rid of the separate CQobjEvo for OpenMP or make it
        # pickleable.  The new (regular) CQobjEvo is pickleable itself.
        if "omp" in self.compiled:
            del _dict_['compiled_qobjevo']
        return _dict_

    def __setstate__(self, state):
        self.__dict__ = state
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


def _dynamic_argument_raise(op, state):
    raise TypeError(
        "unknown shape for evolution, evolver type "
        + repr(op.type)
        + " shape "
        + repr(op.shape)
        + ", state shape "
        + repr(state.shape)
    )


def _dynamic_argument(t, op, state, what, e_op):
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


# Function defined inside another function cannot be pickled,
# Using class instead
class _UnitedFuncCaller:
    def __init__(self, funclist, args, dynamics_args, cte):
        self.funclist = funclist
        self.args = args
        self.dynamics_args = dynamics_args
        self.issuper = cte.issuper
        self.dims = cte.dims
        self.shape = cte.shape

    def set_args(self, args, dynamics_args):
        self.args = args
        self.dynamics_args = dynamics_args

    def dyn_args(self, t, state, shape):
        for name, what, e_op in self.dynamics_args:
            self.args[name] = _dynamic_argument(t, self, state, what, e_op)

    def __call__(self, t, args=None):
        now_args = self.args.copy()
        if args is not None:
            now_args.update(args)
        return [func(t, now_args) for func in self.funclist]

    def get_args(self):
        return self.args


class _Norm2():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return self.func(t, args)*np.conj(self.func(t, args))


class _Shift():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return np.conj(self.func(t + args["_t0"], args))


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

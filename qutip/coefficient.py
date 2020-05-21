#cython: language_level=3
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

from qutip.qobj import Qobj
import qutip.settings as qset
from qutip.interpolate import Cubic_Spline
from scipy.interpolate import CubicSpline, interp1d
from functools import partial, wraps
from types import FunctionType, BuiltinFunctionType
import numpy as np
from numbers import Number
from qutip.qobjevo_codegen import (_compile_str_single, _compiled_coeffs,
                                   _compiled_coeffs_python)
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi,
                                 spmv, cy_spmm_tr)
import atexit
import pickle
import sys
import scipy
import os
from re import sub


class _file_list:
    """
    Contain temp a list .pyx to clean
    """
    def __init__(self):
        self.files = []
        self.clean_old()

    def add(self, file_):
        self.files += [file_ + ".pyx"]

    def clean(self):
        import os
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

    def clean_old(self, max_age=qset.max_age):
        # On importing Qutip, remove temp files left-over for more than 1 week.
        import glob
        import time
        root = qset.tmproot
        bases = ["td_Qobj_single_str",
                 "qobjevo_compiled_coeff_",
                 "cqobjevo_compiled_coeff_"]
        for base in bases:
            for file in glob.glob(os.path.join(root, base + "*")):
                try:
                    age = (time.time() - os.path.getctime(file)) / 3600
                    if age > max_age:
                        os.remove(file)
                except:
                    pass
                    # Warn that the  file could not be deleted?

coeff_files = _file_list()
atexit.register(coeff_files.clean)


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)

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
    def __init__(self, tlist, coeff, args=None):
        self.coeff = coeff
        self.tlist = tlist
        try:
            use_step_func = args["_step_func_coeff"]
        except KeyError:
            use_step_func = 0
        if use_step_func:
            self.func = interp1d(
                self.tlist, self.coeff, kind="previous",
                bounds_error=False, fill_value=0.)
        else:
            self.func = CubicSpline(self.tlist, self.coeff)

    def __call__(self, t, args={}):
        return self.func([t])[0]


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, coeff_func):
        self.coeff_func = coeff_func

    def __call__(self, t, args={}):
        return self.coeff_func(t, args["_state_vec"], args)


class Coefficient:
    def __init__(self, base, tlist=None, args={}):
        self._call = None
        self.base = base
        self.type = None
        self.tlist = tlist
        self.args = args

        if isinstance(base, Cubic_Spline):
            self._call = base
            self.type = "spline"
        elif callable(base):
            self._call = base
            self.type = "func"
        elif isinstance(base, str):
            self._call = _StrWrapper(base)
            self.type = "string"
        elif isinstance(op_k[1], np.ndarray):
            if not isinstance(tlist, np.ndarray) or not \
                    len(op_k[1]) == len(tlist):
                raise TypeError("Time list does not match")
            self._call = _CubicSplineWrapper(tlist, base, args=args),
            self.type = "array"

    def _check_old_with_state(self):
        if self.type == "func":
            try:
                op.get_coeff(0., self.args)
            except TypeError as e:
                self._call = _StateAsArgs(self._call)
                op = EvoElement((op.qobj, nfunc, nfunc, "func"))
                add_vec = True
        if add_vec:
            self.dynamics_args += [("_state_vec", "vec", None)]

    def copy(self):
        if self.type == "array":
            return Coefficient(self.base.copy(),
                               tlist=self.tlist,
                               args=self.args)
        else:
            return Coefficient(self.base)

    def __add__(self, other):
        if not isinstance(other, Coefficient):
            raise TypeError
        if self.type == "str" and other.type == "str":
            return Coefficient("(" + self.base + ") + (" + other.base + ")")
        if self.type == "array" and other.type == "array":
            if np.allclose(self.tlist, other.tlist):
                return Coefficient(self.base + other.base,
                                   tlist=self.tlist,
                                   args=self.args)
        return _Add([self, other])

    def __mul__(self, other):
        if not isinstance(other, Coefficient):
            raise TypeError
        if self.type == "str" and other.type == "str":
            return Coefficient("(" + self.base + ") * (" + other.base + ")")
        if self.type == "array" and other.type == "array":
            if np.allclose(self.tlist, other.tlist):
                return Coefficient(self.base * other.base,
                                   tlist=self.tlist,
                                   args=self.args)
        return _Prod([self, other])

    def _cdc(self):
        if op.type == "string":
            return Coefficient("norm(" + self.base + ")")
        elif op.type == "array":
            return = Coefficient(np.abs(self.base)**2,
                                 tlist=self.tlist,
                                 args=self.args)
        else:
            return _Norm2(self)

    def conj(self):
        if op.type == "string":
            return Coefficient("conj(" + self.base + ")")
        elif op.type == "array":
            return = Coefficient(np.conj(self.base)**2,
                                 tlist=self.tlist,
                                 args=self.args)
        else:
            return _Conj(self)

    def _shift(self):
        if op.type == "string":
            return Coefficient("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                               "(t+_t0)", " " + self.base + " ")
        else:
            return _Shift(self)

    def __call__(self, t, args):
        return self._call(t, args)

    def compile(self, use_cython=qset.use_cython, self.args):
        if use_cython:
            self._call, file_ = _compile_str_single(part.coeff, self.args)
        else:
            self._call, file_ = _compile_str_single(part.coeff, self.args)
        coeff_files.add(file_)
        funclist.append(get_coeff)


class _Norm2(Coefficient):
    def __init__(self, f):
        self.func = f
        self.type = "decorated"

    def __call__(self, t, args):
        return self.func(t, args)*np.conj(self.func(t, args))

    def copy():
        return _Norm2(self.func.copy())


class _Shift(Coefficient):
    def __init__(self, f):
        self.func = f
        self.type = "decorated"

    def __call__(self, t, args):
        return np.conj(self.func(t + args["_t0"], args))

    def copy():
        return _Shift(self.func.copy())


class _Conj(Coefficient):
    def __init__(self, f):
        self.func = f
        self.type = "decorated"

    def __call__(self, t, args):
        return np.conj(self.func(t, args))

    def copy():
        return _Conj(self.func.copy())


class _Prod(Coefficient):
    def __init__(self, f, g):
        self.func_1 = f
        self.func_2 = g
        self.type = "decorated"

    def __call__(self, t, args):
        return self.func_1(t, args) * self.func_2(t, args)

    def copy():
        return _Prod(self.func_1.copy(), self.func_2.copy())


class _Add(Coefficient):
    def __init__(self, f, g):
        self.func_1 = f
        self.func_2 = g
        self.type = "decorated"

    def __call__(self, t, args):
        return self.func_1(t, args) + self.func_2(t, args)

    def copy():
        return _Add(self.func_1.copy(), self.func_2.copy())


def compile_coeff(coeffs, args, dyn_args, dims, shape,
                  use_cython=qset.use_cython):
    Code = None
    compilable = all(op.type in ["string", "array", "spline"]
                     for op in coeffs)
    need_compile = any(op.type == "string" for op in coeffs)

    if not compilable and (not need_compile or use_cython):
        for op in coeffs:
            op.compile(use_cython=True)
        coeff_get = _UnitedFuncCaller(coeffs, args, dyn_args, dims, shape)

    elif not compilable:
        _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(coeffs,
                                                                args, dyn_args)
        coeff_files.add(file_)
        self.coeff_get = _UnitedStrCaller(coeffs, args, dyn_args, dims, shape)

    elif all(op.type == "array" for op in coeffs):
        try:
            use_step_func = self.args["_step_func_coeff"]
        except KeyError:
            use_step_func = 0
        tlist = coeffs[0].tlist
        if np.allclose(np.diff(tlist),
                tlist[1] - tlist[0]):
            if use_step_func:
                coeff_get = StepCoeffCte(coeffs, None, tlist)
            else:
                coeff_get = InterCoeffCte(coeffs, None, tlist)
        else:
            if use_step_func:
                coeff_get = StepCoeffT(coeffs, None, tlist)
            else:
                coeff_get = InterCoeffT(coeffs, None, tlist)

    elif all(op.type == "spline" for op in coeffs):
        coeff_get = InterpolateCoeff(coeffs, None, None)

    else
        if use_cython:
            # All factor can be compiled
            coeff_get, Code, file_ = _compiled_coeffs(coeffs, args, dyn_args)
            coeff_files.add(file_)
        else:
            # All factor can be compiled
            _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(coeffs,
                                                                    args,
                                                                    dyn_args)
            coeff_files.add(file_)
            coeff_get = _UnitedStrCaller(coeffs, args, dyn_args, dims, shape)

    return coeff_get, Code


# Function defined inside another function cannot be pickled,
# Using class instead
class _UnitedFuncCaller:
    def __init__(self, funclist, args, dynamics_args, dims, shape):
        self.funclist = funclist
        self.args = args
        self.dynamics_args = dynamics_args
        self.dims = dims
        self.shape = shape

    def set_args(self, args, dynamics_args=[]):
        self.args = args
        if dynamics_args:
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
                elif shape[1] == 1: # ket
                    self.args[name] = Qobj(mat, dims=[self.dims[1],[1]])
                else:  # rho
                    self.args[name] = Qobj(mat, dims=self.dims[1])
            elif what == "expect":
                if shape[1] == op.cte.shape[1]: # same shape as object
                    self.args[name] = op.mul_mat(t, mat).trace()
                else:
                    self.args[name] = op.expect(t, state)

    def __call__(self, t, args={}):
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

    def copy(self):
        res = _UnitedFuncCaller(self.funclist,
                                self.args, self.dynamics_args,
                                self.dims, self.shape)





def _compile_coeff(self):
    Code = None
    if self.type in ["func"]:
        funclist = [part.get_coeff for part in self.ops]
        self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                           self.dynamics_args,
                                           self.cte)
    elif self.type in ["mixed_callable"] and self.use_cython:
        funclist = []
        for part in self.ops:
            if isinstance(part.get_coeff, _StrWrapper):
                get_coeff, file_ = _compile_str_single(part.coeff,
                                                       self.args)
                coeff_files.add(file_)
                funclist.append(get_coeff)
            else:
                funclist.append(part.get_coeff)
            self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                           self.dynamics_args,
                                           self.cte)
    elif self.type in ["mixed_callable"]:
        funclist = [part.get_coeff for part in self.ops]
        _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(
                                                self.ops,
                                                self.args,
                                                self.dynamics_args,
                                                self.tlist)
        coeff_files.add(file_)
        self.coeff_get = _UnitedStrCaller(funclist, self.args,
                                          self.dynamics_args,
                                          self.cte)
    elif self.type in ["string", "mixed_compilable"]:
        if self.use_cython:
            # All factor can be compiled
            self.coeff_get, Code, file_ = _compiled_coeffs(
                                                self.ops,
                                                self.args,
                                                self.dynamics_args,
                                                self.tlist)
            coeff_files.add(file_)
        else:
            # All factor can be compiled
            _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(
                                                self.ops,
                                                self.args,
                                                self.dynamics_args,
                                                self.tlist)
            coeff_files.add(file_)
            funclist = [part.get_coeff for part in self.ops]
            self.coeff_get = _UnitedStrCaller(funclist, self.args,
                                              self.dynamics_args,
                                              self.cte)

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

    elif self.type == "spline":
        self.coeff_get = InterpolateCoeff(self.ops, None, None)

    return Code


def type(coeffs):
    compilable = all(op.type in ["string", "array", "spline"] for op in coeffs)

        self.compiled = ""
        self.compiled_qobjevo = None
        self.coeff_get = None
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

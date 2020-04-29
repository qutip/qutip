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
"""
Function to build cython code from str, compile and import it.
Used by cQobjEvo.
cy/codegen.py does the same thing for specific solver
"""
import os
import numpy as np
from qutip.cy.inter import _prep_cubic_spline
import time
import sys


def _try_remove(filename):
    try:
        os.remove(filename)
    except Exception:
        pass


def _import_str(code, basefilename, obj_name, cythonfile=False):
    """
    Import 'obj_name' defined in 'code'.
    Using a temporary file starting by 'basefilename'.
    """
    filename = (basefilename + str(hash(code))[1:4] +
                str(os.getpid()) + time.strftime("%M%S"))
    tries = 0
    import_list = []
    ext = ".pyx" if cythonfile else ".py"
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    while not import_list and tries < 3:
        try_file = filename + str(tries)
        file_ = open(try_file+ext, "w")
        file_.writelines(code)
        file_.close()
        if not os.access(try_file, os.R_OK):
            time.sleep(0.1)
        codeString = str("from " + try_file +
                         " import " + obj_name + '\n' +
                         "import_list.append(" + obj_name + ")")
        try:
            import_code = compile(codeString, '<string>', 'exec')
            exec(import_code, locals())
        except (ModuleNotFoundError, ImportError) as e:
            time.sleep(0.05)
            tries += 1
            _try_remove(try_file+ext)
            err = e
    if not import_list:
        raise Exception("Could not convert string to importable function, "
                        "tmpfile:" + try_file + ext) from err
    coeff_obj = import_list[0]
    return coeff_obj, try_file + ext


def _compile_str_single(string, args):
    """Create and import a cython compiled function from text
    """
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
    _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

    Code = """#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
import scipy.special as spe
cimport numpy as np
cimport cython
from qutip.cy.math cimport erf, zerf
cdef double pi = 3.14159265358979323
include """+_include_string+"""

@cython.boundscheck(False)
@cython.wraparound(False)
def f(double t, args):
"""
    for name, value in args.items():
        if name in string:
            Code += "    " + name + " = args['" + name + "']\n"
    Code += "    return " + string + "\n"

    return _import_str(Code, "td_Qobj_single_str", "f", True)


def _compiled_coeffs(ops, args, dyn_args, tlist):
    """Create and import a cython compiled class for coeff that
    need compilation.
    """
    code = _make_code_4_cimport(ops, args, dyn_args, tlist)
    coeff_obj, filename = _import_str(code, "cqobjevo_compiled_coeff_",
                                      "CompiledStrCoeff", True)
    return coeff_obj(ops, args, tlist, dyn_args), code, filename


def _make_code_4_cimport(ops, args, dyn_args, tlist):
    """
    Create the code for a CoeffFunc cython class the wraps
    the string coefficients, array_like coefficients and Cubic_Spline.
    """
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

    code = """#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
import scipy.special as spe
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.inter cimport _spline_complex_t_second, _spline_complex_cte_second
from qutip.cy.inter cimport _spline_float_t_second, _spline_float_cte_second
from qutip.cy.inter cimport _step_float_cte, _step_complex_cte
from qutip.cy.inter cimport _step_float_t, _step_complex_t
from qutip.cy.interpolate cimport (interp, zinterp)
from qutip.cy.cqobjevo_factor cimport StrCoeff
from qutip.cy.cqobjevo cimport CQobjEvo
from qutip.cy.math cimport erf, zerf
from qutip.qobj import Qobj
cdef double pi = 3.14159265358979323

include """ + _include_string + "\n\n"

    compile_list = []
    N_np = 0

    for op in ops:
        if op.type == "string":
            compile_list.append(op.coeff)

        elif op.type == "array":
            spline, dt_cte = _prep_cubic_spline(op[2], tlist)
            t_str = "_tlist"
            y_str = "_array_" + str(N_np)
            s_str = "_spline_" + str(N_np)
            N_times = str(len(tlist))
            dt_times = str(tlist[1]-tlist[0])
            try:
                use_step_func = args["_step_func_coeff"]
            except KeyError:
                use_step_func = 0
            if dt_cte:
                if isinstance(op.coeff[0], (float, np.float32, np.float64)):
                    if use_step_func:
                        string = "_step_float_cte(t, " + t_str + ", " +\
                                y_str + ", " + N_times + ")"
                    else:
                        string = "_spline_float_cte_second(t, " + t_str + ", " +\
                                y_str + ", " + s_str + ", " + N_times + ", " +\
                                dt_times + ")"

                elif isinstance(op.coeff[0], (complex, np.complex128)):
                    if use_step_func:
                        string = "_step_complex_cte(t, " + t_str + ", " +\
                                y_str + ", " + N_times + ")"
                    else:
                        string = "_spline_complex_cte_second(t, " + t_str + ", " +\
                                y_str + ", " + s_str + ", " + N_times + ", " +\
                                dt_times + ")"
            else:
                if isinstance(op.coeff[0], (float, np.float32, np.float64)):
                    if use_step_func:
                        string = "_step_float_t(t, " + t_str + ", " +\
                             y_str + ", " + N_times + ")"
                    else:
                        string = "_spline_float_t_second(t, " + t_str + ", " +\
                             y_str + ", " + s_str + ", " + N_times + ")"
                elif isinstance(op.coeff[0], (complex, np.complex128)):
                    if use_step_func:
                        string = "_step_complex_t(t, " + t_str + ", " +\
                             y_str + ", " + N_times + ")"
                    else:
                        string = "_spline_complex_t_second(t, " + t_str + ", " +\
                             y_str + ", " + s_str + ", " + N_times + ")"
            compile_list.append(string)
            args[t_str] = tlist
            args[y_str] = op.coeff
            args[s_str] = spline
            N_np += 1

        elif op.type == "spline":
            y_str = "_array_" + str(N_np)
            if op[1].is_complex:
                string = "zinterp(t, _CSstart, _CSend, " + y_str + ")"
            else:
                string = "interp(t, _CSstart, _CSend, " + y_str + ")"
            compile_list.append(string)
            args["_CSstart"] = op.coeff.a
            args["_CSend"] = op.coeff.b
            args[y_str] = op.coeff.coeffs
            N_np += 1

    code += "cdef class CompiledStrCoeff(StrCoeff):\n"
    normal_args = args.copy()

    for name, _, _ in dyn_args:
        del normal_args[name]

    for name, value in normal_args.items():
        if not isinstance(name, str):
            raise Exception("All arguments key must be string " +
                            "and valid variables name")
        if isinstance(value, np.ndarray) and \
                isinstance(value[0], (float, np.float32, np.float64)):
            code += "    cdef double[::1] " + name + "\n"
        elif isinstance(value, np.ndarray) and \
                isinstance(value[0], (complex, np.complex128)):
            code += "    cdef complex[::1] " + name + "\n"
        elif isinstance(value, (complex, np.complex128)):
            code += "    cdef complex " + name + "\n"
        elif np.isscalar(value):
            code += "    cdef double " + name + "\n"
        else:
            code += "    cdef object " + name + "\n"

    code += "\n"
    if normal_args:
        code += "    def set_args(self, args):\n"
        for name, value in normal_args.items():
            code += "        self." + name + "=args['" + name + "']\n"
        code += "\n"
    code += "    cdef void _call_core(self, double t, complex * coeff):\n"

    for name, value in normal_args.items():
        if isinstance(value, np.ndarray) and \
                isinstance(value[0], (float, np.float32, np.float64)):
            code += "        cdef double[::1] " + name + " = self." +\
                    name + "\n"
        elif isinstance(value, np.ndarray) and \
                isinstance(value[0], (complex, np.complex128)):
            code += "        cdef complex[::1] " + name + " = self." +\
                    name + "\n"
        elif isinstance(value, (complex, np.complex128)):
            code += "        cdef complex " + name + " = self." + name + "\n"
        elif np.isscalar(value):
            code += "        cdef double " + name + " = self." + name + "\n"
        else:
            code += "        cdef object " + name + " = self." + name + "\n"

    expect_i = 0
    for name, what, op in dyn_args:
        if what == "vec":
            code += "        cdef complex[::1] " + name + " = self._vec\n"
        if what == "mat":
            code += "        cdef np.ndarray[complex, ndim=2] " + name + \
                    " = np.array(self._vec).reshape(" \
                    "(self._mat_shape[0], self._mat_shape[1]), order='F')\n"
        if what == "Qobj":
            # ToDo: Use cython dense to fast_sparse
            code += "        " + name + " = Qobj(np.array(self._vec).reshape(" \
                    "(self._mat_shape[0], self._mat_shape[1]), order='F'))\n"
        if what == "expect":
            code += "        cdef complex " + name + " = self._expect_vec[" \
                    + str(expect_i) + "]\n"
            expect_i += 1

    code += "\n"
    for i, str_coeff in enumerate(compile_list):
        code += "        coeff[" + str(i) + "] = " + str_coeff + "\n"

    return code


def _compiled_coeffs_python(ops, args, dyn_args, tlist):
    """Create and import a cython compiled class for coeff that
    need compilation.
    """
    code = _make_code_4_python_import(ops, args, dyn_args, tlist)
    coeff_obj, filename = _import_str(code, "qobjevo_compiled_coeff_",
                                      "_UnitedStrCaller", False)
    return coeff_obj, code, filename


code_python_pre = """
# This file is generated automatically by QuTiP.
import numpy as np
import scipy.special as spe
import scipy
from qutip.qobjevo import _UnitedFuncCaller

def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)

sin = np.sin
cos = np.cos
tan = np.tan
asin = np.arcsin
acos = np.arccos
atan = np.arctan
pi = np.pi
sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
asinh = np.arcsinh
acosh = np.arccosh
atanh = np.arctanh
exp = np.exp
log = np.log
log10 = np.log10
erf = scipy.special.erf
zerf = scipy.special.erf
sqrt = np.sqrt
real = np.real
imag = np.imag
conj = np.conj
abs = np.abs
norm = lambda x: np.abs(x)**2
arg = np.angle

class _UnitedStrCaller(_UnitedFuncCaller):
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

    def __call__(self, t, args={}):
        if args:
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        out = []

"""

code_python_post = """
        return out

    def get_args(self):
        return self.args

"""


def _make_code_4_python_import(ops, args, dyn_args, tlist):
    code = code_python_pre
    for key in args:
        code += "        " + key + " = now_args['" + key + "']\n"

    for i, op in enumerate(ops):
        if op.type == "string":
            code += "        out.append(" + op.coeff + ")\n"
        else:
            code += "        out.append(self.funclist[" + str(i) + \
                    "](t, now_args))\n"
    code += code_python_post
    return code

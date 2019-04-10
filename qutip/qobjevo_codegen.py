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
import numpy as np
from qutip.cy.inter import _prep_cubic_spline


def _compile_str_single(string, args):
    """Create and import a cython compiled function from text
    """
    import os
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
    filename = "td_Qobj_single_str" + str(hash(Code))[1:]
    file = open(filename+".pyx", "w")
    file.writelines(Code)
    file.close()
    str_func = []
    import_code = compile('from ' + filename + ' import f\n'
                          "str_func.append(f)",
                          '<string>', 'exec')
    exec(import_code, locals())

    try:
        os.remove(filename+".pyx")
    except:
        pass

    return str_func[0], filename


def _compiled_coeffs(ops, args, dyn_args, tlist):
    """Create and import a cython compiled class for coeff that
    need compilation.
    """
    code = _make_code_4_cimport(ops, args, dyn_args, tlist)
    filename = "cqobjevo_compiled_coeff_"+str(hash(code))[1:]

    file = open(filename+".pyx", "w")
    file.writelines(code)
    file.close()
    import_list = []

    import_code = compile('from ' + filename + ' import CompiledStrCoeff\n' +
                          "import_list.append(CompiledStrCoeff)",
                          '<string>', 'exec')
    exec(import_code, locals())
    coeff_obj = import_list[0](ops, args, tlist, dyn_args)

    try:
        os.remove(filename+".pyx")
    except:
        pass

    return coeff_obj, code, filename


def _make_code_4_cimport(ops, args, dyn_args, tlist):
    """
    Create the code for a CoeffFunc cython class the wraps
    the string coefficients, array_like coefficients and Cubic_Spline.
    """
    import os
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
            if dt_cte:
                if isinstance(op.coeff[0], (float, np.float32, np.float64)):
                    string = "_spline_float_cte_second(t, " + t_str + ", " +\
                              y_str + ", " + s_str + ", " + N_times + ", " +\
                              dt_times + ")"
                elif isinstance(op.coeff[0], (complex, np.complex128)):
                    string = "_spline_complex_cte_second(t, " + t_str + ", " +\
                              y_str + ", " + s_str + ", " + N_times + ", " +\
                              dt_times + ")"
            else:
                if isinstance(op.coeff[0], (float, np.float32, np.float64)):
                    string = "_spline_float_t_second(t, " + t_str + ", " +\
                             y_str + ", " + s_str + ", " + N_times + ")"
                elif isinstance(op.coeff[0], (complex, np.complex128)):
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

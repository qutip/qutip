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
import numpy as np
from qutip.cy.inter import prep_cubic_spline


def _compile_str_single(compile_list):
    import os
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
    _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

    all_str = ""
    for op in compile_list:
        all_str += op[0]
    filename = "td_Qobj_"+str(hash(all_str))[1:]

    Code = """
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

include """+_include_string+"\n"

    for str_coeff in compile_list:
        Code += _str_2_code(str_coeff)

    file = open(filename+".pyx", "w")
    file.writelines(Code)
    file.close()
    str_func = []
    for i in range(len(compile_list)):
        func_name = '_str_factor_' + str(i)
        import_code = compile('from ' + filename + ' import ' + func_name +
                              "\nstr_func.append(" + func_name + ")",
                              '<string>', 'exec')
        exec(import_code, locals())

    try:
        os.remove(filename+".pyx")
    except:
        pass

    return str_func


def _str_2_code(str_coeff):

    func_name = '_str_factor_' + str(str_coeff[2])

    Code = """

@cython.boundscheck(False)
@cython.wraparound(False)

def """ + func_name + "(double t"
    # used arguments
    Code += _get_arg_str(str_coeff[1])
    Code += "):\n"
    Code += "    return " + str_coeff[0] + "\n"

    return Code


def _get_arg_str(args):
    if len(args) == 0:
        return ''

    ret = ''
    for name, value in args.items():
        if isinstance(value, np.ndarray):
            ret += ",\n        np.ndarray[np.%s_t, ndim=1] %s" % \
                (value.dtype.name, name)
        else:
            if isinstance(value, (int, np.int32, np.int64)):
                kind = 'int'
            elif isinstance(value, (float, np.float32, np.float64)):
                kind = 'float'
            elif isinstance(value, (complex, np.complex128)):
                kind = 'complex'
            ret += ",\n        " + kind + " " + name
    return ret


def make_united_f_ptr(ops, args, tlist, return_code=False):
    """Create a cython function which return the coefficients of the
time-dependent parts of an Qobj.
string coefficients are compiled
array_like coefficients become cubic spline (see qutip.cy.inter.pyx)

??? I don't know if it will be useful, but the code is written in a way
    it would be easy to have different tlist for each array_like coefficients
    """
    import os
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

    compile_list = []
    N_np = 0
    args_np = []
    spline_list = []

    for op in ops:
        if op[3] == 2:
            compile_list.append(op[2])
        elif op[3] == 3:
            spline, dt_cte = prep_cubic_spline(op[2], tlist)
            spline_list.append(spline)

            t_str = "str_tlist_" + str(N_np) +", "
            y_str = "str_array_" + str(N_np) +", "
            s_str = "str_spline_" + str(N_np) +", "
            N_times = str(len(tlist))
            dt_times = str(tlist[1]-tlist[0])
            if dt_cte:
                if isinstance(op[2][0], (float, np.float32, np.float64)):
                    string = "spline_float_cte_second(t, " + t_str +\
                             y_str + s_str + N_times + ", " + dt_times + ")"
                    args_np += [[0,dt_cte]]
                elif isinstance(op[2][0], (complex, np.complex128)):
                    string = "spline_complex_cte_second(t, " + t_str +\
                             y_str + s_str + N_times + ", " + dt_times + ")"
                    args_np += [[1,dt_cte]]
            else:
                if isinstance(op[2][0], (float, np.float32, np.float64)):
                    string = "spline_float_t_second(t, " + t_str +\
                             y_str + s_str + N_times + ")"
                    args_np += [[0,dt_cte]]
                elif isinstance(op[2][0], (complex, np.complex128)):
                    string = "spline_complex_t_second(t, " + t_str +\
                             y_str + s_str + N_times + ")"
                    args_np += [[1,dt_cte]]

            compile_list.append(string)
            N_np += 1
    all_str = "" + str(np.random.random())
    for op in compile_list:
        all_str += op[0]
    filename = "td_Qobj_f_ptr"+str(hash(all_str))[1:]

    code = """
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.inter cimport spline_complex_t_second, spline_complex_cte_second
from qutip.cy.inter cimport spline_float_t_second, spline_float_cte_second
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

include """ + _include_string + "\n\n"

    code += "cdef class coeff_args:\n"
    for i, iscplx in enumerate(args_np):
        if iscplx[0]:
            code += "    cdef double[::1] str_tlist_" + str(i) + "\n"
            code += "    cdef complex[::1] str_array_" + str(i) + "\n"
            code += "    cdef complex[::1] str_spline_" + str(i) + "\n"
        else:
            code += "    cdef double[::1] str_tlist_" + str(i) + "\n"
            code += "    cdef double[::1] str_array_" + str(i) + "\n"
            code += "    cdef double[::1] str_spline_" + str(i) + "\n"

    for name, value in args.items():
        if not isinstance(name, str):
            raise Exception("All arguments key must be string " +
                            "and valid variables name")
        if isinstance(value, (int, np.int32, np.int64)):
            code += "    cdef int " + name + "\n"
        elif isinstance(value, (float, np.float32, np.float64)):
            code += "    cdef double " + name + "\n"
        elif isinstance(value, (complex, np.complex128)):
            code += "    cdef complex " + name + "\n"

    code += "\n"
    code += "    def set_array_imag(self, int N, double[::1] tlist, " +\
            "complex[::1] array, complex[::1] spline):\n"
    code += "        if N == -1:\n"
    code += "            pass\n"
    for i, iscplx in enumerate(args_np):
        if iscplx[0]:
            code += "        elif N == " + str(i) + ":\n"
            code += "            self.str_tlist_" + str(i) + "= tlist\n"
            code += "            self.str_array_" + str(i) + "= array\n"
            code += "            self.str_spline_" + str(i) + "= spline\n"
    code += "        else:\n"
    code += "            raise Exception('Bad classification imag')\n"
    code += "\n"
    code += "    def set_array_real(self, int N, double[::1] tlist, " +\
            "double[::1] array, double[::1] spline):\n"
    code += "        if N ==-1:\n"
    code += "            pass\n"
    for i, iscplx in enumerate(args_np):
        if not iscplx[0]:
            code += "        elif N == " + str(i) + ":\n"
            code += "            self.str_tlist_" + str(i) + "= tlist\n"
            code += "            self.str_array_" + str(i) + "= array\n"
            code += "            self.str_spline_" + str(i) + "= spline\n"
    code += "        else:\n"
    code += "            raise Exception('Bad classification real')\n"
    code += "\n"
    code += "    def set_args(self, args):\n"
    if not args:
        code += "        pass\n"
    else:
        for name, value in args.items():
            code += "        self." + name + " = args['" + name + "']\n"
    code += "\n"
    code += "cdef coeff_args np_obj = coeff_args()"
    code += "\n\n"

    code += "cdef void coeff(double t, complex* out):\n"
    if N_np:
        #code += "    cdef int N_times = " + str(len(tlist)) + "\n"
        #code += "    cdef double dt_times = " + str(tlist[1]-tlist[0]) + "\n"
        for i, iscplx in enumerate(args_np):
            if iscplx[0]:
                code += "    cdef double[::1] str_tlist_" + str(i) +\
                        "= np_obj.str_tlist_" + str(i) + "\n"
                code += "    cdef complex[::1] str_array_" + str(i) +\
                        "= np_obj.str_array_" + str(i) + "\n"
                code += "    cdef complex[::1] str_spline_" + str(i) +\
                        "= np_obj.str_spline_" + str(i) + "\n"
            else:
                code += "    cdef double[::1] str_tlist_" + str(i) +\
                        "= np_obj.str_tlist_" + str(i) + "\n"
                code += "    cdef double[::1] str_array_" + str(i) +\
                        "= np_obj.str_array_" + str(i) + "\n"
                code += "    cdef double[::1] str_spline_" + str(i) +\
                        "= np_obj.str_spline_" + str(i) + "\n"
    for name, value in args.items():
        if not isinstance(name, str):
            raise Exception("All arguments key must be string and" +
                            " valid variables name")
        if isinstance(value, (int, np.int32, np.int64)):
            code += "    cdef int " + name + " = np_obj." + name + "\n"
        elif isinstance(value, (float, np.float32, np.float64)):
            code += "    cdef double " + name + " = np_obj." + name + "\n"
        elif isinstance(value, (complex, np.complex128)):
            code += "    cdef complex " + name + " = np_obj." + name + "\n"
    code += "\n"
    for i, str_coeff in enumerate(compile_list):
        code += "    out[" + str(i) + "] = " + str_coeff + "\n"

    code += """

def get_ptr(set_np_obj = False):
    if set_np_obj:
        return np_obj
    else:
        return PyLong_FromVoidPtr(<void*> coeff)
"""

    file = open(filename+".pyx", "w")
    file.writelines(code)
    file.close()
    compile_f_ptr = []

    import_code = compile('from ' + filename + ' import get_ptr' +
                          "\ncompile_f_ptr.append(get_ptr)",
                          '<string>', 'exec')
    exec(import_code, locals())

    np_obj = compile_f_ptr[0](set_np_obj=True)
    if N_np:
        n_op = 0
        for op in ops:
            if op[3] == 3:
                if isinstance(op[2][0], (float, np.float32, np.float64)):
                    op[4] = spline_list[n_op]
                    np_obj.set_array_real(n_op, tlist, op[2], spline_list[n_op])
                elif isinstance(op[2][0], (complex, np.complex128)):
                    op[4] = spline_list[n_op]
                    np_obj.set_array_imag(n_op, tlist, op[2], spline_list[n_op])
                n_op += 1
    if args:
        np_obj.set_args(args)

    try:
        os.remove(filename+".pyx")
    except:
        pass

    if return_code:
        return compile_f_ptr[0], code
    else:
        return compile_f_ptr[0]

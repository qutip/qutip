import numpy as np


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
from qutip.cy.interpolate cimport interp, zinterp
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


def _td_array_to_str(self, op_np2, times):
    """
    Wrap numpy-array based time-dependence in the string-based
    time dependence format
    """
    n = 0
    str_op = []
    np_args = {}

    for op in op_np2:
        td_array_name = "_td_array_%d" % n
        H_td_str = '(0 if (t > %f) else %s[int(round(%d * (t/%f)))])' %\
            (times[-1], td_array_name, len(times) - 1, times[-1])
        np_args[td_array_name] = op[1]
        str_op.append([op[0], H_td_str])
        n += 1

    return str_op, np_args


def make_united_f_ptr(ops, args, tlist, return_code=False):
    import os
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
    _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

    compile_list = []
    N_np = 0
    args_np = []
    for op in ops:
        if op[3] == 2:
            compile_list.append(op[2])
        elif op[3] == 3:
            if isinstance(op[2][0], (float, np.float32, np.float64)):
                string = "interpolate(t, &str_array_" + str(N_np) +\
                         "[0], N_times, dt_times)"
                args_np += [0]
            elif isinstance(op[2][0], (complex, np.complex128)):
                string = "zinterpolate(t, &str_array_" + str(N_np) +\
                         "[0], N_times, dt_times)"
                args_np += [1]
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
from qutip.cy.inter cimport zinterpolate, interpolate
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

include """ + _include_string + "\n\n"

    code += "cdef class coeff_args:\n"
    for i, iscplx in enumerate(args_np):
        if iscplx:
            code += "    cdef complex[::1] str_array_" + str(i) + "\n"
        else:
            code += "    cdef double[::1] str_array_" + str(i) + "\n"

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
    code += "    def set_array_imag(self, int N, " +\
            "np.ndarray[complex, ndim=1] array):\n"
    code += "        if N ==-1:\n"
    code += "            pass\n"
    for i, iscplx in enumerate(args_np):
        if iscplx:
            code += "        elif N ==" + str(i) + ":\n"
            code += "            self.str_array_" + str(i) + "= array\n"
    code += "        else:\n"
    code += "            raise Exception('Bad classification imag')\n"
    code += "\n"
    code += "    def set_array_real(self, int N, " +\
            "np.ndarray[double, ndim=1] array):\n"
    code += "        if N ==-1:\n"
    code += "            pass\n"
    for i, iscplx in enumerate(args_np):
        if not iscplx:
            code += "        elif N ==" + str(i) + ":\n"
            code += "            self.str_array_" + str(i) + "= array\n"
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
        code += "    cdef int N_times = " + str(len(tlist)) + "\n"
        code += "    cdef double dt_times = " + str(tlist[1]-tlist[0]) + "\n"
        for i, iscplx in enumerate(args_np):
            if iscplx:
                code += "    cdef complex[::1] str_array_" + str(i) +\
                        "= np_obj.str_array_" + str(i) + "\n"
            else:
                code += "    cdef double[::1] str_array_" + str(i) +\
                        "= np_obj.str_array_" + str(i) + "\n"
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
                    np_obj.set_array_real(n_op, op[2])
                elif isinstance(op[2][0], (complex, np.complex128)):
                    np_obj.set_array_imag(n_op, op[2])
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

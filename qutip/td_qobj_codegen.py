
import numpy as np


def _compile_str_single(compile_list, args):
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
        Code += _str_2_code(str_coeff, args)

    file = open(filename+".pyx", "w")
    file.writelines(Code)
    file.close()
    str_func = []
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

def _str_2_code(str_coeff, args):

    func_name = '_str_factor_' + str(str_coeff[1])

    Code = """

@cython.boundscheck(False)
@cython.wraparound(False)

def """ + func_name + "(double t"
    #used arguments
    Code += _get_arg_str(args)
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

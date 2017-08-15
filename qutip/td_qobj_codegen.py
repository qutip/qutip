import numpy as np

global _list_of_complied_td_Qobj_
_list_of_complied_td_Qobj_ = []

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
    #used arguments
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

def td_qobj_codegen(obj, return_code=False):
    import os
    code, str_args = make_code(obj)
    all_str = "compiled_td_Qobj_"
    for op in obj.ops:
        all_str += str(op[2])
    filename = "compiled_td_Qobj_"+str(hash(all_str))[1:] + \
                str(np.random.randint(0,1000))

    file = open(filename+".pyx", "w")
    file.writelines(code)
    file.close()
    local = False
    try:
        if local:
            compiled_obj_container = [None]
            compiled_obj_ptr = [None]
            import_code = compile('from ' + filename + ' import get_object, get_ptr\n' +
                                  'compiled_obj_container[0] = get_object()\n' +
                                  'compiled_obj_ptr[0] = get_ptr()',
                                  '<string>', 'exec')
            exec(import_code, locals())
            compiled_Qobj = compiled_obj_container[0]
            compiled_Qobj.set_data(obj.cte, obj.ops)
            compiled_Qobj.set_args(obj.args, str_args, obj.tlist)
            ptr = compiled_obj_ptr[0]
        else:
            global _list_of_complied_td_Qobj_
            import_code = compile('import ' + filename + '\n' +
                              '_list_of_complied_td_Qobj_.append(' + filename + ')\n',
                              '<string>', 'exec')
            exec(import_code, globals())
            compiled_Qobj = _list_of_complied_td_Qobj_[-1].get_object()
            compiled_Qobj.set_data(obj.cte, obj.ops)
            compiled_Qobj.set_args(obj.args, str_args, obj.tlist)
            ptr = _list_of_complied_td_Qobj_[-1].get_ptr()

    except Exception as e:
        compiled_Qobj = None
        ptr = None
        print("Not compiled")
        print(str(e))

    try:
        os.remove(filename+".pyx")
    except:
        pass

    if return_code:
        return compiled_Qobj, ptr, code
    else:
        return compiled_Qobj, ptr



def make_code(obj):

    import os
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
    _include_string_1 = "'"+_cython_path + "/cy/complex_math.pxi'"
    _include_string_2 = "'"+_cython_path + "/cy/sparse_routines.pxi'"

    ops_cdef, ops_set = make_ops(obj)
    args_cdef, args_set, factor_code, str_args = make_args(obj)
    call_code = make_call(obj)
    expect_code = make_expect(obj.cte.issuper)

    code = \
"""# distutils: language = c++
import numpy as np
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.inter cimport zinterpolate, interpolate
from qutip.cy.spmatfuncs cimport spmvpy
""" + \
    "\ninclude " + _include_string_1 + \
    "\ninclude " + _include_string_2 + \
"""

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef class cy_compiled_td_qobj:
    cdef int total_elem
    cdef int shape0, shape1

    #pointer to data""" + \
    ops_cdef + "    #args\n" + args_cdef + """
    def __init__(self):
        pass

    """ + ops_set + args_set + factor_code + call_code + """
    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        init_CSR(&out, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out)
        scipy_obj = CSR_to_scipy(&out)
        #free_CSR(&out)? data are own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1., out, self.shape0)
        free_CSR(&out_mat)

    def rhs(self, double t, np.ndarray[complex, ndim=1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0, dtype=complex)
        self._rhs_mat(t, &vec[0], &out[0])
        return out

""" + expect_code + """
    def expect(self, double t, np.ndarray[complex, ndim=1] vec, int isherm):
        return self._expect_mat(t, &vec[0], isherm)

cdef cy_compiled_td_qobj ctdqo = cy_compiled_td_qobj()

def get_object():
    return ctdqo

cdef void rhs_mat(double t, complex* vec, complex* out):
    ctdqo._rhs_mat(t, vec, out)

cdef complex expect_mat(double t, complex* vec, int isherm):
    return ctdqo._expect_mat(t, vec, isherm)

def get_ptr():
    cdef void * ptr1 = <void*>rhs_mat
    cdef void * ptr2 = <void*>expect_mat
    return PyLong_FromVoidPtr(ptr1), PyLong_FromVoidPtr(ptr2)"""

    return code, str_args

def make_ops(obj):
    ops_cdef = """
    cdef CSR_Matrix cte"""

    ops_set = """
    def set_data(self, cte, ops):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]

        self.cte = CSR_from_scipy(cte.data)
        cummulative_op = cte.data"""
    if len(obj.ops) >=1:
        for i in range(len(obj.ops)):
            i_str = str(i)
            ops_cdef += "\n    cdef CSR_Matrix op" + i_str
            ops_cdef += "\n    cdef int op" + i_str + "_sum_elem"

            ops_set += "\n        self.op" + i_str + " = CSR_from_scipy(ops[" + i_str + "][0].data)"
            ops_set += "\n        cummulative_op += ops[" + i_str + "][0].data"
            ops_set += "\n        self.op" + i_str + "_sum_elem = cummulative_op.data.shape[0]"
        ops_cdef += "\n"
        ops_set += "\n\n        self.total_elem = self.op" + i_str + "_sum_elem\n\n"
    else:
        ops_set += "\n\n        self.total_elem = cummulative_op.data.shape[0]\n\n"
    return ops_cdef, ops_set


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



def make_args(obj):
    args = obj.args
    args_cdef = ""

    args_set = """
    def set_args(self, args, str_args, tlist):\n"""

    factor_code = """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void factor(self, t, complex* out):\n"""

    if obj.tlist is not None:
        args_cdef += "    cdef double dt\n"
        args_cdef += "    cdef int N\n"
        args_set += "        self.dt = tlist[-1] / (tlist.shape[0]-1)\n"
        args_set += "        self.N = tlist.shape[0]\n"
        factor_code += "        cdef int N = self.N\n"
        factor_code += "        cdef double dt = self.dt\n"


    str_args = {}
    for i, op in enumerate(obj.ops):
        if op[3] == 3:
            i_str = str(i)
            str_args["str_array_" + i_str] = op[2]

    for name, value in args.items():
        if isinstance(value, (int, np.int32, np.int64)):
            args_cdef += "    cdef int " + name + "\n"
            args_set += "        self." + name + " = args['" + name +"']\n"
            factor_code += "        cdef int " + name + " = self." + name + "\n"
        elif isinstance(value, (float, np.float32, np.float64)):
            args_cdef += "    cdef double " + name + "\n"
            args_set += "        self." + name + " = args['" + name +"']\n"
            factor_code += "        cdef double " + name + " = self." + name + "\n"
        elif isinstance(value, (complex, np.complex128)):
            args_cdef += "    cdef complex " + name + "\n"
            args_set += "        self." + name + " = args['" + name +"']\n"
            factor_code += "        cdef complex " + name + " = self." + name + "\n"

    for i, (name, value) in enumerate(str_args.items()):
        v0 = value[0]
        i_str = str(i)
        if isinstance(v0, (float, np.float32, np.float64)):
            args_cdef += "    cdef double * " + name + "\n"
            args_set += "        cdef np.ndarray[double, ndim=1] str_" + i_str + " = str_args['" + name + "']\n"
            args_set += "        self." + name + " = &str_" + i_str + "[0]\n"
            factor_code += "        cdef double* " + name + " = self." + name + "\n"
        elif isinstance(v0, (complex, np.complex128)):
            args_cdef += "    cdef complex * " + name + "\n"
            args_set += "        cdef np.ndarray[complex, ndim=1] str_" + i_str + " = str_args['" + name + "']\n"
            args_set += "        self." + name + " = &str_" + i_str + "[0]\n"
            factor_code += "        cdef complex* " + name + " = self." + name + "\n"

    if args_set[-2] == ":":
        args_set += "        pass\n"
    if factor_code[-2] == ":":
        factor_code += "        pass\n"

    args_cdef += "\n"
    args_set += "\n"
    factor_code += "\n"

    for i, op in enumerate(obj.ops):
        i_str = str(i)
        if op[3] == 2:
            factor_code += "        out[" + i_str + "] = " + op[2] + "\n"
        if op[3] == 3:
            v0 = op[2][0]
            if isinstance(v0, (float, np.float32, np.float64)):
                factor_code += "        out[" + i_str + "] = interpolate(t, str_array_" + i_str + ", N, dt)\n"
            elif isinstance(v0, (complex, np.complex128)):
                factor_code += "        out[" + i_str + "] = zinterpolate(t, str_array_" + i_str + ", N, dt)\n"

    factor_code += "\n"

    return args_cdef, args_set, factor_code, str_args

def make_call(obj):
    N = len(obj.ops)
    call_code = """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, CSR_Matrix * out):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(""" + str(N) + """, dtype=complex)
        self.factor(t, &coeff[0])
    """

    if N == 0:
        call_code += """
        free_CSR(out)
        copy_CSR(out, &self.cte)"""

    elif N == 1:
        call_code += """
        _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                       self.op0.data, self.op0.indices, self.op0.indptr, coeff[0],
                       out, self.shape0, self.shape1)"""
    elif N == 2:
        call_code += """
        cdef CSR_Matrix cummulative_0
        init_CSR(&cummulative_0, self.op0_sum_elem, self.shape0, self.shape1)
        self.factor(t, &coeff[0])
        _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                       self.op0.data, self.op0.indices, self.op0.indptr, coeff[0],
                       &cummulative_0, self.shape0, self.shape1)

        _zcsr_add_core(cummulative_0.data, cummulative_0.indices, cummulative_0.indptr,
                       self.op1.data, self.op1.indices, self.op1.indptr, coeff[1],
                       out, self.shape0, self.shape1)\n"""
        call_code += "        free_CSR(&cummulative_0)\n"
    else:
        call_code += """
        cdef CSR_Matrix cummulative_0
        init_CSR(&cummulative_0, self.op0_sum_elem, self.shape0, self.shape1)
        self.factor(t, &coeff[0])
        _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                       self.op0.data, self.op0.indices, self.op0.indptr, coeff[0],
                       &cummulative_0, self.shape0, self.shape1)\n"""

        for i in range(1,N-1):
            i_str = str(i)
            i_s_p = str(i-1)
            call_code += "        cdef CSR_Matrix cummulative_" + i_str + "\n"
            call_code += "        init_CSR(&cummulative_" + i_str + ", self.op" + i_str + "_sum_elem, self.shape0, self.shape1)\n"
            call_code += "        _zcsr_add_core(cummulative_" + i_s_p + ".data, cummulative_" + i_s_p + ".indices, cummulative_" + i_s_p + ".indptr,\n"
            call_code += "                       self.op" + i_str + ".data, self.op" + i_str + ".indices, self.op" + i_str + ".indptr, coeff[" + i_str + "],\n"
            call_code += "                       &cummulative_" + i_str + ", self.shape0, self.shape1)\n"
            call_code += "        free_CSR(&cummulative_" + i_s_p + ")\n"

        i_str = str(N-1)
        i_s_p = str(N-2)
        call_code += "        _zcsr_add_core(cummulative_" + i_s_p + ".data, cummulative_" + i_s_p + ".indices, cummulative_" + i_s_p + ".indptr,\n"
        call_code += "                       self.op" + i_str + ".data, self.op" + i_str + ".indices, self.op" + i_str + ".indptr, coeff[" + i_str + "],\n"
        call_code += "                       out, self.shape0, self.shape1)\n\n"
        call_code += "        free_CSR(&cummulative_" + i_s_p + ")\n"
    call_code += "\n\n"
    return call_code

def make_expect(super):
    if not super:
        return """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        cdef np.ndarray[complex, ndim=1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1., &y[0], self.shape0)
        cdef int row
        cdef complex dot = 0
        free_CSR(&out_mat)

        for row from 0 <= row < self.shape0:
            dot += conj(vec[row])*y[row]

        if isherm:
            return real(dot)
        else:
            return dot
"""

    else:
        return """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)

        cdef int row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>np.sqrt(num_rows)
        cdef complex dot = 0.0

        for row from 0 <= row < num_rows by n+1:
            row_start = out_mat.indptr[row]
            row_end = out_mat.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += out_mat.data[jj]*vec[out_mat.indices[jj]]
        free_CSR(&out_mat)

        if isherm:
            return real(dot)
        else:
            return dot
"""

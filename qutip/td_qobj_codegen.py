
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

    for str_coeff, args in compile_list:
        Code += _str_2_code(str_coeff)

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


def _str_2_code(str_coeff):

    func_name = '_str_factor_' + str(str_coeff[1])

    Code = """

@cython.boundscheck(False)
@cython.wraparound(False)

def """ + func_name + "(double t"
    #used arguments
    Code += _get_arg_str(str_coeff[2])
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

def td_qobj_codegen(obj):

    import os
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
    _include_string_1 = "'"+_cython_path + "/cy/complex_math.pxi'"
    _include_string_2 = "'"+_cython_path + "/cy/sparse_routines.pxi'"

    ops_cdef, ops_set = make_ops(obj)
    args_cdef, args_set = make_args(obj)
    factor_call_code = make_factor(obj)

    code = \
"""import numpy as np
cimport numpy as cnp
import cython
from qutip import qobj
from scipy import sparse.csr_matrix as csr
from qutip.cy.spmath import _zcsr_add_core
from qutip.cy.inter import zinterpolate, interpolate
""" + \
    "\ninclude " + _include_string_1 + \
    "\ninclude " + _include_string_2 + \
"""

cdef extern from "src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef void split_qobj(object obj, complex*, int*, int*):
    cdef cnp.ndarray[complex, ndim=1] data = obj.data.data
    cdef cnp.ndarray[complex, ndim=1] ptr = obj.data.indptr
    cdef cnp.ndarray[complex, ndim=1] ind = obj.data.indices
    return &ptr[0], &ind[0], &data[0]

cdef class cy_compiled_td_qobj:
    cdef int total_elem
    cdef int shape0, shape1

    #pointer to data""" + \
    ops_cdef + "    #args\n" + args_cdef + """
    def __init__(self):
        pass

    """ + ops_set + args_set + factor_call_code + """
        def call(self, double t, bool data=False):
            cdef CSR_Matrix * out
            out = new CSR_Matrix()
            init_CSR(out, self.total_elem, self.shape0, self.shape1)
            self._c_call(t, out)
            scipy_obj = CSR_to_scipy(out)
            if data:
                return scipy_obj
            else:
                return qobj(scipy_obj)


        @cython.boundscheck(False)
        @cython.wraparound(False)
        @cython.cdivision(True)
        cdef void _rhs_mat(self, double t, complex* vec, complex* out):
            cdef CSR_Matrix * out_mat
            out_mat = new CSR_Matrix()
            init_CSR(out_mat, self.total_elem, self.shape0, self.shape1)
            self._call_core(t, out_mat)
            zspmvpy(out_mat.data, out_mat.ind, out_mat.ptr, vec, 1., out, self.shape0)

        def rhs(self, double t, np.ndarray[complex, ndim=1] vec):
            cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0)
            self._rhs_mat(t, vec, out)
            return out

        def rhs_ptr():
            void * ptr = <void*>self._rhs_mat
            return PyLong_FromVoidPtr(ptr)


        @cython.boundscheck(False)
        @cython.wraparound(False)
        @cython.cdivision(True)
        cdef complex _expect_mat(self, double t, complex* vec, int herm):
            cdef CSR_Matrix * out_mat
            out_mat = new CSR_Matrix()
            init_CSR(out_mat, self.total_elem, self.shape0, self.shape1)
            self._call_core(t, out_mat)
            return cy_expect_psi_csr(out_mat.data, out_mat.ind, out_mat.ptr, vec, herm)

        def expect(self, double t, np.ndarray[complex, ndim=1] vec, int isherm):
            return self._expect_mat(t, vec, isherm)

        def expect_ptr():
            void * ptr = <void*>self._expect_mat
            return PyLong_FromVoidPtr(ptr)

    """

def make_ops(obj):
    ops_cdef = """
    cdef CSR_Matrix cte_obj
    cdef CSR_Matrix * cte"""

    ops_set = """
    def set_data(self, cte, ops):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]

        self.cte_obj = CSR_from_scipy(cte)
        self.cte = &self.cte_obj
        cummulative_op = cte"""

    for i in range(len(obj.ops)):
        i_str = str(i)
        ops_cdef += "\n    cdef CSR_Matrix op" + i_str + "_obj"
        ops_cdef += "\n    cdef CSR_Matrix * op" + i_str + ""
        ops_cdef += "\n    cdef int op" + i_str + "_sum_elem"

        ops_set += "\n        self.op" + i_str + "_obj = CSR_from_scipy(ops[" + i_str + "][0])"
        ops_set += "\n        self.op" + i_str + " = &self.op" + i_str + "_obj"
        ops_set += "\n        cummulative_op += ops[" + i_str + "][0]"
        ops_set += "\n        op" + i_str + "_sum_elem = cummulative_op.data.shape[0]"

    ops_set += "\n\n        total_elem = op" + i_str + "_sum_elem"
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



def make_ops(obj):
    args = obj.args
    if len(args) == 0:
        args_cdef = ""

    args_set = """
    def set_args(self, args, tlist):\n"""

    factor_code = """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void factor(self, t, complex* out):"""

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
            args_set += "         self." + name + " = args['" + name +"']\n"
            factor_code += "        cdef int " + name + " = self." + name + "\n"
        elif isinstance(value, (float, np.float32, np.float64)):
            args_cdef += "    cdef double " + name + "\n"
            args_set += "         self." + name + " = args['" + name +"']\n"
            factor_code += "        cdef double " + name + " = self." + name + "\n"
        elif isinstance(value, (complex, np.complex128)):
            args_cdef += "    cdef complex " + name + "\n"
            args_set += "         self." + name + " = args['" + name +"']\n"
            factor_code += "        cdef complex " + name + " = self." + name + "\n"

    for i, (name, value) in enumerate(str_args.items()):
        v0 = value[0]
        i_str = str(i)
        if isinstance(v0, (float, np.float32, np.float64)):
            args_cdef += "    cdef double * " + name + "\n"
            args_set += "         cdef np.ndarray[double, ndim=1] str_" + i_str + " = str_args['" + name + "']"
            args_set += "         self." + name + " = &str_" + i_str + "[0]
            factor_code += "        cdef double* " + name + " = self." + name + "\n"
        elif isinstance(v0, (complex, np.complex128)):
            args_cdef += "    cdef complex * " + name + "\n"
            args_set += "         cdef np.ndarray[complex, ndim=1] str_" + i_str + " = str_args['" + name + "']"
            args_set += "         self." + name + " = &str_" + i_str + "[0]
            factor_code += "        cdef complex* " + name + " = self." + name + "\n"

    args_cdef += "\n"
    args_set += "\n\n"
    factor_code += "\n"

    for i, op in enumerate(obj.ops):
        i_str = str(i)
        if op[3] == 2:
            factor_code += "        factor[" + i_str + "] = " + op[2] + "\n"
        if op[3] == 3:
            v0 = op[2][0]
            if isinstance(v0, (float, np.float32, np.float64)):
                factor_code += "        factor[" + i_str + "] = interpolate(t, str_array_ + i_str + , N, dt)\n"
            elif isinstance(v0, (complex, np.complex128)):
                factor_code += "        factor[" + i_str + "] = zinterpolate(t, str_array_ + i_str + , N, dt)\n"

    factor_code += "\n"

    return args_cdef, args_set, factor_code

def make_call(obj):
    N = len(obj.ops)
    call_code = """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, CSR_Matrix * out):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(""" +str(N)+""")
        self.factor(t, coeff)
    """

    if N == 0:
        out = self.cte

    elif N == 1:
        call_code += """
        _zcsr_add_core(cte_obj.data, cte_obj.ind, cte_obj.ptr,
                       op0_obj.ptr, op0_obj.ptr, op0_obj.data, coeff[0]
                       out, self.shape0, self.shape1)"""
    elif N == 2:
        call_code += """
        cdef CSR_Matrix * cummulative_0
        cummulative_0 = new CSR_Matrix()
        init_CSR(cummulative_0, self.op0_sum_elem, self.shape0, self.shape1)
        self.factor(t, coeff)
        _zcsr_add_core(cte_obj.data, cte_obj.ind, cte_obj.ptr,
                       op0_obj.data, op0_obj.ind, op0_obj.ptr, coeff[0]
                       cummulative_0, self.shape0, self.shape1)

        _zcsr_add_core(cummulative_0[0].data, cummulative_0[0].ind, cummulative_0[0].ptr,
                       op1_obj.ptr, op1_obj.ptr, op1_obj.data, coeff[1]
                       out, self.shape0, self.shape1)"""
    else:

    for i in range(N-1):
        i_str = str(i)
        call_code += """
        cdef CSR_Matrix * cummulative_0
        cummulative_0 = new CSR_Matrix()
        init_CSR(cummulative_0, self.op0_sum_elem, self.shape0, self.shape1)
        self.factor(t, coeff)
        _zcsr_add_core(cte_obj.data, cte_obj.ind, cte_obj.ptr,
                       op0_obj.data, op0_obj.ind, op0_obj.ptr, coeff[0]
                       cummulative_0, self.shape0, self.shape1)

        _zcsr_add_core(cummulative_0[0].data, cummulative_0[0].ind, cummulative_0[0].ptr,
                       op1_obj.ptr, op1_obj.ptr, op1_obj.data, coeff[1]
                       out, self.shape0, self.shape1)"""
#    factor_call_code = make_factor(obj)

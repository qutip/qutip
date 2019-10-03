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
import os
import numpy as np
from qutip.interpolate import Cubic_Spline
_cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
_include_string = "'"+_cython_path+"/complex_math.pxi'"
__all__ = ['Codegen']


class Codegen():
    """
    Class for generating cython code files at runtime.
    """
    def __init__(self, h_terms=None, h_tdterms=None, h_td_inds=None,
                 args=None, c_terms=None, c_tdterms=[], c_td_inds=None,
                 c_td_splines=[], c_td_spline_flags=[],
                 type='me', config=None,
                 use_openmp=False, omp_components=None, omp_threads=None):
        import sys
        import os
        sys.path.append(os.getcwd())

        # Hamiltonian time-depdendent pieces
        self.type = type
        if isinstance(h_terms, int):
            h_terms = range(h_terms)

        self.h_terms = h_terms  # number of H pieces
        self.h_tdterms = h_tdterms  # list of time-dependent strings
        self.h_td_inds = h_td_inds  # indicies of time-dependnt terms
        self.args = args  # args for strings

        # Collapse operator time-depdendent pieces
        self.c_terms = c_terms  # number of C pieces
        self.c_tdterms = c_tdterms  # list of time-dependent strings
        self.c_td_inds = c_td_inds  # indicies of time-dependent terms
        self.c_td_splines = c_td_splines #List of c_op spline arrays
        self.c_td_spline_flags = c_td_spline_flags #flags for oper or super

        # Code generator properties
        self.code = []  # strings to be written to file
        self.level = 0  # indent level
        self.config = config

        #openmp settings
        self.use_openmp = use_openmp
        self.omp_components = omp_components
        self.omp_threads = omp_threads

    def write(self, string):
        """write lines of code to self.code"""
        self.code.append("    " * self.level + string + "\n")

    def file(self, filename):
        """open file called filename for writing"""
        self.file = open(filename, "w")

    def generate(self, filename="rhs.pyx"):
        """generate the file"""
        for line in cython_preamble(self.use_openmp):
            self.write(line)

        # write function for Hamiltonian terms (there is always at least one
        # term)
        for line in cython_checks() + self.ODE_func_header():
            self.write(line)
        self.indent()
        for line in self.func_vars():
            self.write(line)
        self.write(self.func_end())
        self.dedent()

        # generate collapse operator functions if any c_terms
        if any(self.c_tdterms):
            for line in (cython_checks() + self.col_spmv_header() +
                         cython_col_spmv()):
                self.write(line)
            self.indent()
            for line in self.func_which():
                self.write(line)
            self.write(self.func_end())
            self.dedent()
            for line in (cython_checks() + self.col_expect_header() +
                         cython_col_expect(self.args)):
                self.write(line)
            self.indent()
            for line in self.func_which_expect():
                self.write(line)
            self.write(self.func_end_real())
            self.dedent()

        self.file(filename)
        self.file.writelines(self.code)
        self.file.close()
        self.config.cgen_num += 1

    def indent(self):
        """increase indention level by one"""
        self.level += 1

    def dedent(self):
        """decrease indention level by one"""
        if self.level == 0:
            raise SyntaxError("Error in code generator")
        self.level -= 1

    def _get_arg_str(self, args):
        if len(args) == 0:
            return ''

        ret = ''
        for name, value in self.args.items():
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
                #kind = type(value).__name__
                ret += ",\n        " + kind + " " + name
        return ret

    def ODE_func_header(self):
        """Creates function header for time-dependent ODE RHS."""
        func_name = "def cy_td_ode_rhs("
        # strings for time and vector variables
        input_vars = ("\n        double t" +
                      ",\n        complex[::1] vec")
        for k in self.h_terms:
            input_vars += (",\n        " +
                           "complex[::1] data%d," % k +
                           "int[::1] idx%d," % k +
                           "int[::1] ptr%d" % k)

        kk = len(self.h_tdterms)
        for jj in range(len(self.c_td_splines)):
            input_vars += (",\n        " +
                           "complex[::1] data%d," % (jj+kk) +
                           "int[::1] idx%d," % (jj+kk) +
                           "int[::1] ptr%d" % (jj+kk))

        if any(self.c_tdterms):
            for k in range(len(self.h_terms),
                           len(self.h_terms) + len(self.c_tdterms)):
                input_vars += (",\n        " +
                               "complex[::1] data%d," % k +
                               "int[::1] idx%d," % k +
                               "int[::1] ptr%d" % k)

        #Add array for each Cubic_Spline term
        spline = 0
        for htd in (self.h_tdterms+self.c_td_splines):
            if isinstance(htd, Cubic_Spline):
                if not htd.is_complex:
                    input_vars += (",\n        " +
                                   "double[::1] spline%d" % spline)
                else:
                    input_vars += (",\n        " +
                                   "complex[::1] spline%d" % spline)
                spline += 1

        input_vars += self._get_arg_str(self.args)
        func_end = "):"
        return [func_name + input_vars + func_end]

    def col_spmv_header(self):
        """
        Creates function header for time-dependent
        collapse operator terms.
        """
        func_name = "def col_spmv("
        input_vars = ("int which, double t, complex[::1] " +
                      "data, int[::1] idx, int[::1] " +
                      "ptr, complex[::1] vec")
        input_vars += self._get_arg_str(self.args)
        func_end = "):"
        return [func_name + input_vars + func_end]

    def col_expect_header(self):
        """
        Creates function header for time-dependent
        collapse expectation values.
        """
        func_name = "def col_expect("
        input_vars = ("int which, double t, complex[::1] " +
                      "data, int[::1] idx, int[::1] " +
                      "ptr, complex[::1] vec")
        input_vars += self._get_arg_str(self.args)
        func_end = "):"
        return [func_name + input_vars + func_end]

    def func_vars(self):
        """Writes the variables and their types & spmv parts"""
        func_vars = ["", 'cdef size_t row', 'cdef unsigned int num_rows = vec.shape[0]',
                     "cdef double complex * " +
                     'out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))']
        func_vars.append(" ")
        tdterms = self.h_tdterms
        hinds = 0
        spline = 0
        for ht in self.h_terms:
            hstr = str(ht)
            # Monte-carlo evolution
            if self.type == 'mc':
                if ht in self.h_td_inds:
                    if isinstance(tdterms[hinds], str):
                        td_str= tdterms[hinds]
                    elif isinstance(tdterms[hinds], Cubic_Spline):
                        S = tdterms[hinds]
                        if not S.is_complex:
                            td_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                        else:
                            td_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                        spline += 1
                    hinds += 1
                else:
                    td_str = "1.0"
                str_out = "spmvpy(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, &out[0], num_rows)" % (
                        ht, ht, ht, td_str)
                func_vars.append(str_out)
            # Master and Schrodinger evolution
            else:
                if self.h_tdterms[ht] == "1.0":
                    if self.use_openmp and self.omp_components[ht]:
                        str_out = "spmvpy_openmp(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], 1.0, out, num_rows, %s)" % (
                            ht, ht, ht, self.omp_threads)
                    else:
                        str_out = "spmvpy(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], 1.0, out, num_rows)" % (
                                ht, ht, ht)
                else:
                    if isinstance(self.h_tdterms[ht], str):
                        if self.use_openmp and self.omp_components[ht]:
                            str_out = "spmvpy_openmp(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows, %s)" % (
                                ht, ht, ht, self.h_tdterms[ht], self.omp_threads)
                        else:
                            str_out = "spmvpy(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows)" % (
                                    ht, ht, ht, self.h_tdterms[ht])
                    elif isinstance(self.h_tdterms[ht], Cubic_Spline):
                        S = self.h_tdterms[ht]
                        if not S.is_complex:
                            interp_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                        else:
                            interp_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                        spline += 1
                        if self.use_openmp and self.omp_components[ht]:
                            str_out = "spmvpy_openmp(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows, %s)" % (
                                ht, ht, ht, interp_str, self.omp_threads)
                        else:
                            str_out = "spmvpy(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows)" % (
                                ht, ht, ht, interp_str)
                    #Do nothing if not a specified type
                    else:
                        str_out=  ''
                func_vars.append(str_out)

        cstr = 0
        if len(self.c_tdterms) > 0:
            # add a spacer line between Hamiltonian components and collapse
            # components.
            func_vars.append(" ")
            terms = len(self.c_tdterms)
            tdterms = self.c_tdterms
            cinds = 0
            for ct in range(terms):
                cstr = str(ct + hinds + 1)
                str_out = "spmvpy(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows)" % (
                        cstr, cstr, cstr, " (" + tdterms[ct] + ")**2")
                cinds += 1
                func_vars.append(str_out)

        #Collapse operators have cubic spline td-coeffs
        if len(self.c_td_splines) > 0:
            func_vars.append(" ")
            for ct in range(len(self.c_td_splines)):
                S = self.c_td_splines[ct]
                c_idx = self.c_td_spline_flags[ct]
                if not S.is_complex:
                    interp_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                else:
                    interp_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                spline += 1

                #check if need to wrap string with ()**2
                if c_idx > 0:
                    interp_str = "("+interp_str+")**2"
                c_idx = abs(c_idx)
                if self.use_openmp and self.omp_components[ht]:
                    str_out = "spmvpy_openmp(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows, %s)" % (
                        c_idx, c_idx, c_idx, interp_str, self.omp_threads)
                else:
                    str_out = "spmvpy(&data%s[0], &idx%s[0], &ptr%s[0], &vec[0], %s, out, num_rows)" % (
                        c_idx, c_idx, c_idx, interp_str)
                func_vars.append(str_out)


        return func_vars

    def func_which(self):
        """Writes 'else-if' statements forcollapse operator eval function"""
        out_string = []
        ind = 0
        out_string.append("cdef size_t kk")
        out_string.append("cdef complex ctd = %s" % self.c_tdterms[ind])
        for k in self.c_td_inds:
            out_string.append("if which == " + str(k) + ":")
            out_string.append("""\
    for kk in range(num_rows):
            out[kk] *= ctd
                     """)
            ind += 1
        return out_string

    def func_which_expect(self):
        """Writes 'else-if' statements for collapse expect function
        """
        out_string = []
        ind = 0
        for k in self.c_td_inds:
            out_string.append("if which == " + str(k) + ":")
            out_string.append("    out *= conj(" + self.c_tdterms[ind] + ")")
            ind += 1
        return out_string

    def func_end(self):
        return """\
cdef np.npy_intp dims = num_rows
    cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_COMPLEX128, out)
    PyArray_ENABLEFLAGS(arr_out, np.NPY_OWNDATA)
    return arr_out
"""

    def func_end_real(self):
        return "return real(out)"


def cython_preamble(use_openmp=False):
    """
    Returns list of code segments for Cython preamble.
    """
    if use_openmp:
        openmp_string='from qutip.cy.openmp.parfuncs cimport spmvpy_openmp'
    else:
        openmp_string=''

    return ["""#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.
# (C) 2011 and later, QuSTaR

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
"""
+openmp_string+
"""
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf, zerf
cdef double pi = 3.14159265358979323

include """+_include_string+"""

"""]


def cython_checks():
    """
    List of strings that turn off Cython checks.
    """
    return ["""
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)"""]


def cython_col_spmv():
    """
    Writes col_SPMV vars.
    """
    return ["""\
    cdef size_t row
    cdef unsigned int jj, row_start, row_end
    cdef unsigned int num_rows = vec.shape[0]
    cdef complex dot
    cdef complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))

    for row in range(num_rows):
        dot = 0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            dot = dot + data[jj] * vec[idx[jj]]
        out[row] = dot
    """]


def cython_col_expect(args):
    """
    Writes col_expect vars.
    """
    return ["""\
    cdef size_t row
    cdef int num_rows = vec.shape[0]
    cdef complex out = 0.0
    cdef np.ndarray[complex, ndim=1, mode='c'] dot = col_spmv(which, t, data, idx, ptr,
                                                    vec%s)

    for row in range(num_rows):
        out += conj(vec[row]) * dot[row]
    """ % "".join(["," + str(td_const[0])
                   for td_const in args.items()]) if args else ""]

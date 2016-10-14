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
                 type='me', config=None):
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

        # Code generator properties
        self.code = []  # strings to be written to file
        self.level = 0  # indent level
        self.config = config

    def write(self, string):
        """write lines of code to self.code"""
        self.code.append("    " * self.level + string + "\n")

    def file(self, filename):
        """open file called filename for writing"""
        self.file = open(filename, "w")

    def generate(self, filename="rhs.pyx"):
        """generate the file"""
        for line in cython_preamble():
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
                      ",\n        np.ndarray[CTYPE_t, ndim=1] vec")
        for k in self.h_terms:
            input_vars += (",\n        " +
                           "np.ndarray[CTYPE_t, ndim=1] data%d," % k +
                           "np.ndarray[int, ndim=1] idx%d," % k +
                           "np.ndarray[int, ndim=1] ptr%d" % k)
        if any(self.c_tdterms):
            for k in range(len(self.h_terms),
                           len(self.h_terms) + len(self.c_tdterms)):
                input_vars += (",\n        " +
                               "np.ndarray[CTYPE_t, ndim=1] data%d," % k +
                               "np.ndarray[int, ndim=1] idx%d," % k +
                               "np.ndarray[int, ndim=1] ptr%d" % k)
        
        #Add array for each Cubic_Spline term
        spline = 0
        for htd in self.h_tdterms:
            if isinstance(htd, Cubic_Spline):
                if not htd.is_complex:
                    input_vars += (",\n        " +
                                   "np.ndarray[DTYPE_t, ndim=1] spline%d" % spline)
                else:
                    input_vars += (",\n        " +
                                   "np.ndarray[CTYPE_t, ndim=1] spline%d" % spline)
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
        input_vars = ("int which, double t, np.ndarray[CTYPE_t, ndim=1] " +
                      "data, np.ndarray[int] idx,np.ndarray[int] " +
                      "ptr,np.ndarray[CTYPE_t, ndim=1] vec")
        input_vars += self._get_arg_str(self.args)
        func_end = "):"
        return [func_name + input_vars + func_end]

    def col_expect_header(self):
        """
        Creates function header for time-dependent
        collapse expectation values.
        """
        func_name = "def col_expect("
        input_vars = ("int which, double t, np.ndarray[CTYPE_t, ndim=1] " +
                      "data, np.ndarray[int] idx,np.ndarray[int] " +
                      "ptr,np.ndarray[CTYPE_t, ndim=1] vec")
        input_vars += self._get_arg_str(self.args)
        func_end = "):"
        return [func_name + input_vars + func_end]

    def func_vars(self):
        """Writes the variables and their types & spmv parts"""
        func_vars = ["", 'cdef Py_ssize_t row', 'cdef int num_rows = len(vec)',
                     'cdef np.ndarray[CTYPE_t, ndim=1] ' +
                     'out = np.zeros((num_rows),dtype=np.complex)']
        func_vars.append(" ")
        tdterms = self.h_tdterms
        hinds = 0
        spline = 0
        for ht in self.h_terms:
            hstr = str(ht)
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
                str_out = "spmvpy(data%s, idx%s, ptr%s, vec, %s, out)" % (
                        ht, ht, ht, td_str)
                func_vars.append(str_out)
            else:
                if self.h_tdterms[ht] == "1.0":
                    str_out = "spmvpy(data%s, idx%s, ptr%s, vec, 1.0, out)" % (
                        ht, ht, ht)
                else:
                    if isinstance(self.h_tdterms[ht], str):
                        str_out = "spmvpy(data%s, idx%s, ptr%s, vec, %s, out)" % (
                            ht, ht, ht, self.h_tdterms[ht])
                    elif isinstance(self.h_tdterms[ht], Cubic_Spline):
                        S = self.h_tdterms[ht]
                        if not S.is_complex:
                            interp_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                        else:
                            interp_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, spline)
                        spline += 1
                        str_out = "spmvpy(data%s, idx%s, ptr%s, vec, %s, out)" % (
                            ht, ht, ht, interp_str)
                func_vars.append(str_out)

        if len(self.c_tdterms) > 0:
            # add a spacer line between Hamiltonian components and collapse
            # components.
            func_vars.append(" ")
            terms = len(self.c_tdterms)
            tdterms = self.c_tdterms
            cinds = 0
            for ct in range(terms):
                cstr = str(ct + hinds + 1)
                str_out = "spmvpy(data%s, idx%s, ptr%s, vec, %s, out)" % (
                        cstr, cstr, cstr, " abs(" + tdterms[ct] + ")**2")
                cinds += 1
                func_vars.append(str_out)
        return func_vars

    def func_which(self):
        """Writes 'else-if' statements forcollapse operator eval function"""
        out_string = []
        ind = 0
        for k in self.c_td_inds:
            out_string.append("if which == " + str(k) + ":")
            out_string.append("    out *= " + self.c_tdterms[ind])
            ind += 1
        return out_string

    def func_which_expect(self):
        """Writes 'else-if' statements for collapse expect function
        """
        out_string = []
        ind = 0
        for k in self.c_td_inds:
            out_string.append("if which == " + str(k) + ":")
            out_string.append("    out *= conj(" +
                              self.c_tdterms[ind] + ")")
            ind += 1
        return out_string

    def func_end(self):
        return "return out"

    def func_end_real(self):
        return "return np.float64(np.real(out))"


def cython_preamble():
    """
    Returns list of code segments for Cython preamble.
    """
    return ["""\
# This file is generated automatically by QuTiP.
# (C) 2011 and later, P. D. Nation & J. R. Johansson

import numpy as np
cimport numpy as np
cimport cython
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
cdef double pi = 3.14159265358979323

include """+_include_string+"""

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t
"""]


def cython_checks():
    """
    List of strings that turn off Cython checks.
    """
    return ["""
@cython.boundscheck(False)
@cython.wraparound(False)"""]


def cython_col_spmv():
    """
    Writes col_SPMV vars.
    """
    return ["""\
    cdef Py_ssize_t row
    cdef int jj, row_start, row_end
    cdef int num_rows = len(vec)
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=1] out = np.zeros(num_rows, dtype=np.complex)

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
    cdef Py_ssize_t row
    cdef int num_rows=len(vec)
    cdef CTYPE_t out = 0.0
    cdef np.ndarray[CTYPE_t, ndim=1] vec_ct = vec.conj()
    cdef np.ndarray[CTYPE_t, ndim=1] dot = col_spmv(which, t, data, idx, ptr,
                                                    vec%s)

    for row in range(num_rows):
        out += vec_ct[row] * dot[row]
    """ % "".join(["," + str(td_const[0])
                   for td_const in args.items()]) if args else ""]

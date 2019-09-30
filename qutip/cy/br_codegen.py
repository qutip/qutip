# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR.
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
import qutip.settings as qset
from qutip.interpolate import Cubic_Spline
_cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
_include_string = "'"+_cython_path+"/complex_math.pxi'"
__all__ = ['BR_Codegen']


class BR_Codegen(object):
    """
    Class for generating Bloch-Redfield time-dependent code
    at runtime.
    """
    def __init__(self, h_terms=None, h_td_terms=None, h_obj=None,
                c_terms=None, c_td_terms=None, c_obj=None,
                a_terms=None, a_td_terms=None,
                spline_count=[0,0],
                coupled_ops=[],
                coupled_lengths=[],
                coupled_spectra=[],
                config=None, sparse=False,
                use_secular=None,
                sec_cutoff=0.1,
                args=None,
                use_openmp=False,
                omp_thresh=None,
                omp_threads=None,
                atol=None):
        import sys
        import os
        sys.path.append(os.getcwd())

        # Hamiltonian time-depdendent pieces
        self.h_terms = h_terms  # number of H pieces
        self.h_td_terms = h_td_terms
        self.h_obj = h_obj
        # Collapse operator time-depdendent pieces
        self.c_terms = c_terms  # number of C pieces
        self.c_td_terms = c_td_terms
        self.c_obj = c_obj
        # BR operator time-depdendent pieces
        self.a_terms = a_terms  # number of A pieces
        self.a_td_terms = a_td_terms
        self.spline_count = spline_count
        self.use_secular = int(use_secular)
        self.sec_cutoff = sec_cutoff
        self.args = args
        self.sparse = sparse
        self.spline = 0
        # Code generator properties
        self.code = []  # strings to be written to file
        self.level = 0  # indent level
        self.config = config
        if atol is None:
            self.atol = qset.atol
        else:
            self.atol = atol

        self.use_openmp = use_openmp
        self.omp_thresh = omp_thresh
        self.omp_threads = omp_threads

        self.coupled_ops = coupled_ops
        self.coupled_lengths = coupled_lengths
        self.coupled_spectra = coupled_spectra

    def write(self, string):
        """write lines of code to self.code"""
        self.code.append("    " * self.level + string + "\n")

    def file(self, filename):
        """open file called filename for writing"""
        self.file = open(filename, "w")

    def generate(self, filename="rhs.pyx"):
        """generate the file"""
        for line in cython_preamble(self.use_openmp)+self.aop_td_funcs():
            self.write(line)

        # write function for Hamiltonian terms (there is always
        # be at least one term)
        for line in cython_checks() + self.ODE_func_header():
            self.write(line)
        self.indent()
        #Reset spline count
        self.spline = 0
        for line in self.func_vars()+self.ham_add_and_eigsolve()+ \
                self.br_matvec_terms()+["\n"]:
            self.write(line)

        for line in self.func_end():
            self.write(line)
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
        for k in range(self.h_terms):
            input_vars += (",\n        " +
                           "complex[::1,:] H%d" % k)

        #Add array for each Cubic_Spline H term
        for htd in self.h_td_terms:
            if isinstance(htd, Cubic_Spline):
                if not htd.is_complex:
                    input_vars += (",\n        " +
                                   "double[::1] spline%d" % self.spline)
                else:
                    input_vars += (",\n        " +
                                   "complex[::1] spline%d" % self.spline)
                self.spline += 1


        for k in range(self.c_terms):
            input_vars += (",\n        " +
                           "complex[::1,:] C%d" % k)

        #Add array for each Cubic_Spline c_op term
        for ctd in self.c_td_terms:
            if isinstance(ctd, Cubic_Spline):
                if not ctd.is_complex:
                    input_vars += (",\n        " +
                                   "double[::1] spline%d" % self.spline)
                else:
                    input_vars += (",\n        " +
                                   "complex[::1] spline%d" % self.spline)
                self.spline += 1


        #Add coupled a_op terms
        for _a in self.a_td_terms:
            if isinstance(_a, Cubic_Spline):
                if not _a.is_complex:
                    input_vars += (",\n        " +
                                   "double[::1] spline%d" % self.spline)
                else:
                    input_vars += (",\n        " +
                                   "complex[::1] spline%d" % self.spline)
                self.spline += 1


        #Add a_op terms
        for k in range(self.a_terms):
            input_vars += (",\n        " +
                           "complex[::1,:] A%d" % k)


        input_vars += (",\n        unsigned int nrows")
        input_vars += self._get_arg_str(self.args)

        func_end = "):"
        return [func_name + input_vars + func_end]

    def func_vars(self):
        """Writes the variables and their types & spmv parts"""
        func_vars = ["", "cdef double complex * " +
                     'out = <complex *>PyDataMem_NEW_ZEROED(nrows**2,sizeof(complex))']
        func_vars.append(" ")
        return func_vars


    def aop_td_funcs(self):
        aop_func_str=[]
        spline_val = self.spline_count[0]
        coupled_val = 0
        kk = 0
        while kk < self.a_terms:
            if kk not in self.coupled_ops:
                aa = self.a_td_terms[kk]
                if isinstance(aa, str):
                    aop_func_str += ["cdef complex spectral{0}(double w, double t): return {1}".format(kk, aa)]
                elif isinstance(aa, tuple):
                    if isinstance(aa[0],str):
                        str0 = aa[0]
                    elif isinstance(aa[0],Cubic_Spline):
                        if not aa[0].is_complex:
                            aop_func_str += ["cdef double[::1] spline{0} = np.array(".format(spline_val)+np.array2string(aa[0].coeffs,separator=',',precision=16)+",dtype=float)"]
                            str0 = "interp(w, %s, %s, spline%s)" % (aa[0].a, aa[0].b, spline_val)
                        else:
                            aop_func_str += ["cdef complex[::1] spline{0} = np.array(".format(spline_val)+np.array2string(aa[0].coeffs,separator=',',precision=16)+",dtype=complex)"]
                            str0 = "zinterp(w, %s, %s, spline%s)" % (aa[0].a, aa[0].b, spline_val)
                        spline_val += 1
                    else:
                        raise Exception('Error parsing tuple.')

                    if isinstance(aa[1],str):
                        str1 = aa[1]
                    elif isinstance(aa[1],Cubic_Spline):
                        if not aa[1].is_complex:
                            aop_func_str += ["cdef double[::1] spline{0} = np.array(".format(spline_val)+np.array2string(aa[1].coeffs,separator=',',precision=16)+",dtype=float)"]
                            str1 = "interp(t, %s, %s, spline%s)" % (aa[1].a, aa[1].b, spline_val)
                        else:
                            aop_func_str += ["cdef complex[::1] spline{0} = np.array(".format(spline_val)+np.array2string(aa[1].coeffs,separator=',',precision=16)+",dtype=complex)"]
                            str1 = "zinterp(t, %s, %s, spline%s)" % (aa[1].a, aa[1].b, spline_val)
                        spline_val += 1
                    else:
                        raise Exception('Error parsing tuple.')

                    aop_func_str += ["cdef complex spectral{0}(double w, double t): return ({1})*({2})".format(kk, str0, str1)]
                else:
                    raise Exception('Invalid a_td_term.')
                kk += 1
            else:
                aa = self.coupled_spectra[coupled_val]
                if isinstance(aa, str):
                    aop_func_str += ["cdef complex spectral{0}(double w, double t): return {1}".format(kk, aa)]
                elif isinstance(aa, Cubic_Spline):
                    if not aa[1].is_complex:
                        aop_func_str += ["cdef double[::1] spline{0} = np.array(".format(spline_val)+np.array2string(aa[1].coeffs,separator=',',precision=16)+",dtype=float)"]
                        str1 = "interp(t, %s, %s, spline%s)" % (aa[1].a, aa[1].b, spline_val)
                    else:
                        aop_func_str += ["cdef complex[::1] spline{0} = np.array(".format(spline_val)+np.array2string(aa[1].coeffs,separator=',',precision=16)+",dtype=complex)"]
                        str1 = "zinterp(t, %s, %s, spline%s)" % (aa[1].a, aa[1].b, spline_val)
                    spline_val += 1
                    aop_func_str += ["cdef complex spectral{0}(double w, double t): return {1}".format(kk, str1)]
                kk += self.coupled_lengths[coupled_val]
                coupled_val += 1

        return aop_func_str


    def ham_add_and_eigsolve(self):
        ham_str = []
        #allocate initial zero-Hamiltonian and eigenvector array in Fortran-order
        ham_str += ['cdef complex[::1, :] H = farray_alloc(nrows)']
        ham_str += ['cdef complex[::1, :] evecs = farray_alloc(nrows)']
        #allocate double array for eigenvalues
        ham_str += ['cdef double * eigvals = <double *>PyDataMem_NEW_ZEROED(nrows,sizeof(double))']
        for kk in range(self.h_terms):
            if isinstance(self.h_td_terms[kk], Cubic_Spline):
                S = self.h_td_terms[kk]
                if not S.is_complex:
                    td_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, self.spline)
                else:
                    td_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, self.spline)
                ham_str += ["dense_add_mult(H, H{0}, {1})".format(kk,td_str)]
                self.spline += 1
            else:
                ham_str += ["dense_add_mult(H, H{0}, {1})".format(kk,self.h_td_terms[kk])]
        #Do the eigensolving
        ham_str += ["ZHEEVR(H, eigvals, evecs, nrows)"]
        #Free H as it is no longer needed
        ham_str += ["PyDataMem_FREE(&H[0,0])"]

        return ham_str

    def br_matvec_terms(self):
        br_str = []
        # Transform vector eigenbasis
        br_str += ["cdef double complex * eig_vec = vec_to_eigbasis(vec, evecs, nrows)"]
        # Do the diagonal liouvillian matvec
        br_str += ["diag_liou_mult(eigvals, eig_vec, out, nrows)"]
        # Do the cop_term matvec for each c_term
        for kk in range(self.c_terms):
            if isinstance(self.c_td_terms[kk], Cubic_Spline):
                S = self.c_td_terms[kk]
                if not S.is_complex:
                    td_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, self.spline)
                else:
                    td_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, self.spline)
                if self.use_openmp:

                    br_str += ["cop_super_mult_openmp(C{0}, evecs, eig_vec, {1}, out, nrows, {2}, {3}, {4})".format(kk,
                                            td_str, self.omp_thresh, self.omp_threads, self.atol)]
                else:
                    br_str += ["cop_super_mult(C{0}, evecs, eig_vec, {1}, out, nrows, {2})".format(kk, td_str, self.atol)]
                self.spline += 1
            else:
                if self.use_openmp:
                    br_str += ["cop_super_mult_openmp(C{0}, evecs, eig_vec, {1}, out, nrows, {2}, {3}, {4})".format(kk,
                                            self.c_td_terms[kk], self.omp_thresh, self.omp_threads, self.atol)]
                else:
                    br_str += ["cop_super_mult(C{0}, evecs, eig_vec, {1}, out, nrows, {2})".format(kk, self.c_td_terms[kk], self.atol)]

        if self.a_terms != 0:
            #Calculate skew and dw_min terms
            br_str += ["cdef double[:,::1] skew = <double[:nrows,:nrows]><double *>PyDataMem_NEW_ZEROED(nrows**2,sizeof(double))"]
            br_str += ["cdef double dw_min = skew_and_dwmin(eigvals, skew, nrows)"]

        #Compute BR term matvec
        kk = 0
        coupled_val = 0
        while kk < self.a_terms:
            if kk not in self.coupled_ops:
                if self.use_openmp:
                    br_str += ["br_term_mult_openmp(t, A{0}, evecs, skew, dw_min, spectral{0}, eig_vec, out, nrows, {1}, {2}, {3}, {4}, {5})".format(kk,
                                        self.use_secular, self.sec_cutoff, self.omp_thresh, self.omp_threads, self.atol)]
                else:
                    br_str += ["br_term_mult(t, A{0}, evecs, skew, dw_min, spectral{0}, eig_vec, out, nrows, {1}, {2}, {3})".format(kk, self.use_secular, self.sec_cutoff, self.atol)]
                kk += 1
            else:
                br_str += ['cdef complex[::1, :] Ac{0} = farray_alloc(nrows)'.format(kk)]
                for nn in range(self.coupled_lengths[coupled_val]):
                    if isinstance(self.a_td_terms[kk+nn], str):
                        br_str += ["dense_add_mult(Ac{0}, A{1}, {2})".format(kk,kk+nn,self.a_td_terms[kk+nn])]
                    elif isinstance(self.a_td_terms[kk+nn], Cubic_Spline):
                        S = self.a_td_terms[kk+nn]
                        if not S.is_complex:
                            td_str = "interp(t, %s, %s, spline%s)" % (S.a, S.b, self.spline)
                        else:
                            td_str = "zinterp(t, %s, %s, spline%s)" % (S.a, S.b, self.spline)
                        br_str += ["dense_add_mult(Ac{0}, A{1}, {2})".format(kk,kk+nn,td_str)]
                    else:
                        raise Exception('Invalid time-dependence fot a_op.')

                if self.use_openmp:
                    br_str += ["br_term_mult_openmp(t, Ac{0}, evecs, skew, dw_min, spectral{0}, eig_vec, out, nrows, {1}, {2}, {3}, {4}, {5})".format(kk,
                                        self.use_secular, self.sec_cutoff, self.omp_thresh, self.omp_threads, self.atol)]
                else:
                    br_str += ["br_term_mult(t, Ac{0}, evecs, skew, dw_min, spectral{0}, eig_vec, out, nrows, {1}, {2}, {3})".format(kk, self.use_secular, self.sec_cutoff, self.atol)]

                br_str += ["PyDataMem_FREE(&Ac{0}[0,0])".format(kk)]
                kk += self.coupled_lengths[coupled_val]
                coupled_val += 1
        return br_str


    def func_end(self):
        end_str = []
        #Transform out vector back to fock basis
        end_str += ["cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = vec_to_fockbasis(out, evecs, nrows)"]
        #Free everything at end
        if self.a_terms != 0:
            end_str += ["PyDataMem_FREE(&skew[0,0])"]
        end_str += ["PyDataMem_FREE(&evecs[0,0])"]
        end_str += ["PyDataMem_FREE(eigvals)"]
        end_str += ["PyDataMem_FREE(eig_vec)"]
        end_str += ["PyDataMem_FREE(out)"]
        end_str += ["return arr_out"]
        return end_str



def cython_preamble(use_omp=False):
    if use_omp:
        call_str = "from qutip.cy.openmp.br_omp cimport (cop_super_mult_openmp, br_term_mult_openmp)"
    else:
        call_str = "from qutip.cy.brtools cimport (cop_super_mult, br_term_mult)"
    """
    Returns list of code segments for Cython preamble.
    """
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
    void PyDataMem_FREE(void * ptr)
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf, zerf
cdef double pi = 3.14159265358979323
from qutip.cy.brtools cimport (dense_add_mult, ZHEEVR, dense_to_eigbasis,
        vec_to_eigbasis, vec_to_fockbasis, skew_and_dwmin,
        diag_liou_mult, spec_func, farray_alloc)
"""
+call_str+
"""
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

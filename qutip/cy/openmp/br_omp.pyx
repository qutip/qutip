#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
from scipy.linalg.cython_blas cimport zgemv
from qutip.cy.spmath cimport (_zcsr_kron_core, _zcsr_kron,
                    _zcsr_add, _zcsr_transpose, _zcsr_adjoint,
                    _zcsr_mult)
from qutip.cy.spconvert cimport fdense2D_to_CSR
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.openmp.parfuncs cimport spmvpy_openmp
from qutip.cy.brtools cimport (spec_func, vec2mat_index, dense_to_eigbasis)
from libc.math cimport fabs, fmin
from libc.float cimport DBL_MAX
from libcpp.vector cimport vector
from qutip.cy.sparse_structs cimport (CSR_Matrix, COO_Matrix)

include "../sparse_routines.pxi"

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double cabs   "abs" (double complex x)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZGEMV(double complex * A, double complex * vec,
                        double complex * out,
                       int Arows, int Acols, int transA = 0,
                       double complex alpha=1, double complex beta=1):
    cdef char tA
    cdef int idx = 1, idy = 1
    if transA == 0:
        tA = b'N'
    elif transA == 1:
        tA = b'T'
    elif transA == 2:
        tA = b'C'
    else:
        raise Exception('Invalid transA value.')
    zgemv(&tA, &Arows, &Acols, &alpha, A, &Arows, vec, &idx, &beta, out, &idy)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cop_super_mult_openmp(complex[::1,:] cop, complex[::1,:] evecs,
                     double complex * vec,
                     double complex alpha,
                     double complex * out,
                     unsigned int nrows,
                     unsigned int omp_thresh,
                     unsigned int nthr,
                     double atol):
    cdef size_t kk
    cdef CSR_Matrix mat1, mat2, mat3, mat4

    cdef complex[::1,:] cop_eig = dense_to_eigbasis(cop, evecs, nrows, atol)

    #Mat1 holds cop_eig in CSR format
    fdense2D_to_CSR(cop_eig, &mat1, nrows, nrows)

    #Multiply by alpha for time-dependence
    for kk in range(mat1.nnz):
        mat1.data[kk] *= alpha

    #Free data associated with cop_eig as it is no longer needed.
    PyDataMem_FREE(&cop_eig[0,0])

    #create temp array of conj data for cop_eig_sparse
    cdef complex * conj_data = <complex *>PyDataMem_NEW(mat1.nnz * sizeof(complex))
    for kk in range(mat1.nnz):
        conj_data[kk] = conj(mat1.data[kk])

    #mat2 holds data for kron(cop.dag(), c)
    init_CSR(&mat2, mat1.nnz**2, mat1.nrows**2, mat1.ncols**2)
    _zcsr_kron_core(conj_data, mat1.indices, mat1.indptr,
                   mat1.data, mat1.indices, mat1.indptr,
                   &mat2,
                   mat1.nrows, mat1.nrows, mat1.ncols)

    #Do spmv with kron(cop.dag(), c)
    if mat2.nnz >= omp_thresh:
        spmvpy_openmp(mat2.data,mat2.indices,mat2.indptr,
            &vec[0], 1, out, nrows**2, nthr)
    else:
        spmvpy(mat2.data,mat2.indices,mat2.indptr,
                &vec[0], 1, out, nrows**2)

    #Free temp conj_data array
    PyDataMem_FREE(conj_data)
    #Free mat2
    free_CSR(&mat2)

    #Create identity in mat3
    identity_CSR(&mat3, nrows)

    #Take adjoint of cop (mat1) -> mat2
    _zcsr_adjoint(&mat1, &mat2)

    #multiply cop.dag() * c (cdc) -> mat4
    _zcsr_mult(&mat2, &mat1, &mat4)

    #Free mat1 and mat2
    free_CSR(&mat1)
    free_CSR(&mat2)

    # kron(eye, cdc) -> mat1
    _zcsr_kron(&mat3, &mat4, &mat1)

    #Do spmv with -0.5*kron(eye, cdc)
    if mat1.nnz >= omp_thresh:
        spmvpy_openmp(mat1.data,mat1.indices,mat1.indptr,
            vec, -0.5, &out[0], nrows**2, nthr)
    else:
        spmvpy(mat1.data,mat1.indices,mat1.indptr,
            vec, -0.5, &out[0], nrows**2)

    #Free mat1 (mat1 and mat2 are currently free)
    free_CSR(&mat1)

    #Take traspose of cdc (mat4) -> mat1
    _zcsr_transpose(&mat4, &mat1)

    #Free mat4 (mat2 and mat4 currently free)
    free_CSR(&mat4)

    # kron(cdct, eye) -> mat2
    _zcsr_kron(&mat1, &mat3, &mat2)

    #Do spmv with -0.5*kron(cdct, eye)
    if mat2.nnz >= omp_thresh:
        spmvpy_openmp(mat2.data,mat2.indices,mat2.indptr,
            vec, -0.5, &out[0], nrows**2, nthr)
    else:
        spmvpy(mat2.data,mat2.indices,mat2.indptr,
            vec, -0.5, &out[0], nrows**2)

    #Free mat1, mat2, and mat3
    free_CSR(&mat1)
    free_CSR(&mat2)
    free_CSR(&mat3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void br_term_mult_openmp(double t, complex[::1,:] A, complex[::1,:] evecs,
                double[:,::1] skew, double dw_min, spec_func spectral,
                double complex * vec, double complex * out,
                unsigned int nrows, int use_secular,
                double sec_cutoff,
                unsigned int omp_thresh,
                unsigned int nthr,
                double atol):

    cdef size_t kk
    cdef size_t I, J # vector index variables
    cdef int[2] ab, cd #matrix indexing variables
    cdef complex[::1,:] A_eig = dense_to_eigbasis(A, evecs, nrows, atol)
    cdef complex elem, ac_elem, bd_elem
    cdef vector[int] coo_rows, coo_cols
    cdef vector[complex] coo_data
    cdef unsigned int nnz
    cdef COO_Matrix coo
    cdef CSR_Matrix csr
    cdef complex[:,::1] non_sec_mat

    for I in range(nrows**2):
        vec2mat_index(nrows, I, ab)
        for J in range(nrows**2):
            vec2mat_index(nrows, J, cd)

            if (not use_secular) or (fabs(skew[ab[0],ab[1]]-skew[cd[0],cd[1]]) < (dw_min * sec_cutoff)):
                elem = (A_eig[ab[0],cd[0]]*A_eig[cd[1],ab[1]]) * 0.5
                elem *= (spectral(skew[cd[0],ab[0]],t)+spectral(skew[cd[1],ab[1]],t))

                if (ab[0]==cd[0]):
                    ac_elem = 0
                    for kk in range(nrows):
                        ac_elem += A_eig[cd[1],kk]*A_eig[kk,ab[1]] * spectral(skew[cd[1],kk],t)
                    elem -= 0.5*ac_elem

                if (ab[1]==cd[1]):
                    bd_elem = 0
                    for kk in range(nrows):
                        bd_elem += A_eig[ab[0],kk]*A_eig[kk,cd[0]] * spectral(skew[cd[0],kk],t)
                    elem -= 0.5*bd_elem

                if (elem != 0):
                    coo_rows.push_back(I)
                    coo_cols.push_back(J)
                    coo_data.push_back(elem)

    PyDataMem_FREE(&A_eig[0,0])

    #Number of elements in BR tensor
    nnz = coo_rows.size()
    coo.nnz = nnz
    coo.rows = coo_rows.data()
    coo.cols = coo_cols.data()
    coo.data = coo_data.data()
    coo.nrows = nrows**2
    coo.ncols = nrows**2
    coo.is_set = 1
    coo.max_length = nnz
    COO_to_CSR(&csr, &coo)
    if csr.nnz > omp_thresh:
        spmvpy_openmp(csr.data, csr.indices, csr.indptr, vec, 1, out, nrows**2, nthr)
    else:
        spmvpy(csr.data, csr.indices, csr.indptr, vec, 1, out, nrows**2)
    free_CSR(&csr)

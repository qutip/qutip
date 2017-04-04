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
include "sparse_routines.pxi"
from scipy.linalg.cython_lapack cimport zheev
from scipy.linalg.cython_blas cimport zgemm
from qutip.cy.spmath cimport (_zcsr_kron_core, _zcsr_kron, 
                    _zcsr_add, _zcsr_transpose, _zcsr_adjoint,
                    _zcsr_mult)
from qutip.cy.spconvert cimport dense2D_to_CSR

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void eigsh(complex[:,::1] H, double[::1] eigvals, int nrows):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned as the rows of H.
    
    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigen values.
    nrows : int
        Number of rows in matrix.
    """
    cdef char jobz = b'V'
    cdef char uplo = b'L'
    cdef int lda = nrows
    cdef int lwork = 18 * nrows
    cdef complex * work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((3*nrows-2) * sizeof(double))
    cdef int info
    
    zheev(&jobz, &uplo, &nrows, &H[0,0], &lda, &eigvals[0], work, &lwork, rwork, &info)
    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")

            
@cython.boundscheck(False)
@cython.wraparound(False)
def liou_from_diag_ham(double[::1] diags):
    cdef unsigned int nrows = diags.shape[0]
    cdef np.ndarray[complex, ndim=1, mode='c'] data = np.empty(nrows**2, dtype=complex)
    cdef np.ndarray[int, ndim=1, mode='c'] ind = np.empty(nrows**2, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] ptr = np.empty(nrows**2+1, dtype=np.int32)
    cdef unsigned int idx, nnz = 0
    cdef size_t ii, jj
    cdef double complex val1, val2, ans
    
    ptr[0] = 0
    for ii in range(nrows):
        val1 = 1j*diags[ii]
        idx = nrows*ii+1 #Here the +1 is to set the next ptr
        for jj in range(nrows):
            val2 = -1j*diags[jj]
            ans = val1 + val2
            if ans != 0:
                data[nnz] = ans
                ind[nnz] = nrows*ii+jj
                ptr[idx+jj] = nnz+1
                nnz += 1
            else:
                ptr[idx+jj] = nnz
    return fast_csr_matrix((data[:nnz],ind[:nnz],ptr), shape=(nrows**2,nrows**2))

    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void liou_diag_ham_mult(double * diags, double complex * vec, 
                        double complex * out, unsigned int nrows) nogil:
    """
    Multiplies a Liouvillian constructed from a diagonal Hamiltonian 
    onto a vectorized density matrix.
    
    Parameters
    ----------
    diags : double ptr
        Pointer to eigvals of Hamiltonian
    vec : complex ptr
        Pointer to density matrix vector
    out : complex ptr
        Pointer to vector storing result
    nrows : int
        Dimension of Hamiltonian.
    """
    cdef unsigned int nnz = 0
    cdef size_t ii, jj
    cdef double complex val, ans
    
    for ii in range(nrows):
        val = 1j*diags[ii]
        for jj in range(nrows):
            ans = val - 1j*diags[jj]
            out[nnz] += ans*vec[nnz]
            nnz += 1
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex * ZGEMM(double complex * A, double complex * B,
         int Arows, int Acols, int Brows, int Bcols,
         int transA = 0, int transB = 0,
         double complex alpha = 1, double complex beta = 0):
    cdef double complex * C = <double complex *>PyDataMem_NEW((Acols*Brows)*sizeof(double complex))
    cdef char tA, tB
    if transA == 0:
        tA = b'N'
    elif transA == 1:
        tA = b'T'
    elif transA == 2:
        tA = b'C'
    else:
        raise Exception('Invalid transA value.')
    if transB == 0:
        tB = b'N'
    elif transB == 1:
        tB = b'T'
    elif transB == 2:
        tB = b'C'
    else:
        raise Exception('Invalid transB value.')
    
    zgemm(&tA, &tB, &Arows, &Bcols, &Brows, &alpha, A, &Arows, B, &Brows, &beta, C, &Arows)
    return C


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef complex[:,::1] dense_to_eigbasis(complex[:, ::1] A, complex[:,::1] evecs):
    cdef int Arows = A.shape[0]
    cdef int Acols = A.shape[1]
    cdef int Brows = evecs.shape[0]
    cdef int Bcols = evecs.shape[1]
    cdef double complex * temp1 = ZGEMM(&A[0,0], &evecs[0,0], 
                                       Arows, Acols, Brows, Bcols, 0, 0)
    cdef double complex * eig_mat = ZGEMM(&evecs[0,0], temp1,
                                       Arows, Acols, Brows, Bcols, 2, 0)
    PyDataMem_FREE(temp1)
    cdef complex[:,::1] mat = <complex[:Acols, :Brows]> eig_mat
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef complex[:,::1] dense_to_fockbasis(complex[:, ::1] A, complex[:,::1] evecs):
    cdef int Arows = A.shape[0]
    cdef int Acols = A.shape[1]
    cdef int Brows = evecs.shape[0]
    cdef int Bcols = evecs.shape[1]
    cdef double complex * temp1 = ZGEMM(&A[0,0], &evecs[0,0], 
                                       Arows, Acols, Brows, Bcols, 0, 2)
    cdef double complex * fock_mat = ZGEMM(&evecs[0,0], temp1,
                                       Arows, Acols, Brows, Bcols, 0, 0)
    PyDataMem_FREE(temp1)
    cdef complex[:,::1] mat = <complex[:Acols, :Brows]> fock_mat
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ham_add_mult(complex[:, ::1] A, 
                  complex[:, ::1] B, 
                  double complex alpha = 1):
    """
    Performs the dense matrix multiplication
    A = A + (alpha*B)
    where A and B are complex 2D square matrices,
    and alpha is a complex coefficient.
    """
    cdef unsigned int nrows = A.shape[0]
    cdef size_t ii, jj
    for ii in range(nrows):
        for jj in range(nrows):
            A[ii,jj] += alpha*B[ii,jj]
            

@cython.boundscheck(False)
@cython.wraparound(False)
def cop_super_term(complex[:,::1] cop, complex[:, ::1] evecs, 
                     complex alpha, unsigned int nrows):
    cdef size_t kk
    cdef complex[:,::1] cop_eig = dense_to_eigbasis(cop, evecs)

    cdef CSR_Matrix mat1, mat2, mat3, mat4, mat5

    dense2D_to_CSR(cop_eig, &mat1, nrows, nrows)
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
    
    #Free temp conj_data array
    PyDataMem_FREE(conj_data)
    #Create identity in mat3
    identity_CSR(&mat3, nrows)
    #Take adjoint cop.H -> mat4
    _zcsr_adjoint(&mat1, &mat4)
    #multiply cop.dag() * c -> mat5
    _zcsr_mult(&mat4, &mat1, &mat5)
    #Free mat1 and mat 4 as we will reuse
    free_CSR(&mat1)
    free_CSR(&mat4)
    # kron(eye, cdc) -> mat1
    _zcsr_kron(&mat3, &mat5, &mat1)
    # Add data from mat2 - 0.5 * cop_sparse -> mat4
    _zcsr_add(&mat2, &mat1, &mat4, -0.5)
    #Free mat1 and mat2 now
    free_CSR(&mat1)
    free_CSR(&mat2)
    #Take traspose of cdc -> mat1
    _zcsr_transpose(&mat5, &mat1)
    free_CSR(&mat5)
    # kron(cdct, eye) -> mat2
    _zcsr_kron(&mat1, &mat3, &mat2)
    free_CSR(&mat3)
    # Add data from mat4 - 0.5 * mat2 -> mat1
    _zcsr_add(&mat4, &mat2, &mat1, -0.5)
    free_CSR(&mat4)
    free_CSR(&mat2)
    return CSR_to_scipy(&mat1)
    
    
    
                       
    
    
    
    
            
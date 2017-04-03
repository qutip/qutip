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
cdef void liou_diag_ham_mult(double * diags, double complex * vec, double complex * out, unsigned int nrows):
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
cpdef np.ndarray[complex, ndim=2, mode='c'] dense_to_eigbasis(complex[:, ::1] A, complex[:,::1] evecs):
    cdef int Arows = A.shape[0]
    cdef int Acols = A.shape[1]
    cdef int Brows = evecs.shape[0]
    cdef int Bcols = evecs.shape[1]
    cdef double complex * temp1 = ZGEMM(&A[0,0], &evecs[0,0], 
                                       Arows, Acols, Brows, Bcols, 0, 0)
    cdef double complex * fock_mat = ZGEMM(&evecs[0,0], temp1,
                                       Arows, Acols, Brows, Bcols, 2, 0)
    PyDataMem_FREE(temp1)
    cdef np.ndarray[complex, ndim=2, mode='c'] mat
    cdef np.npy_intp dims[2] 
    dims[:] = [Acols, Brows]
    mat = np.PyArray_SimpleNewFromData(2, <np.npy_intp *>dims, np.NPY_COMPLEX128, fock_mat)
    PyArray_ENABLEFLAGS(mat, np.NPY_OWNDATA)
    return mat

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[complex, ndim=2, mode='c'] dense_to_fockbasis(complex[:, ::1] A, complex[:,::1] evecs):
    cdef int Arows = A.shape[0]
    cdef int Acols = A.shape[1]
    cdef int Brows = evecs.shape[0]
    cdef int Bcols = evecs.shape[1]
    cdef double complex * temp1 = ZGEMM(&A[0,0], &evecs[0,0], 
                                       Arows, Acols, Brows, Bcols, 0, 2)
    cdef double complex * eig_mat = ZGEMM(&evecs[0,0], temp1,
                                       Arows, Acols, Brows, Bcols, 0, 0)
    PyDataMem_FREE(temp1)
    cdef np.ndarray[complex, ndim=2, mode='c'] mat
    cdef np.npy_intp dims[2] 
    dims[:] = [Acols, Brows]
    mat = np.PyArray_SimpleNewFromData(2, <np.npy_intp *>dims, np.NPY_COMPLEX128, eig_mat)
    PyArray_ENABLEFLAGS(mat, np.NPY_OWNDATA)
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
            
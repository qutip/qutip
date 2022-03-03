#cython: language_level=3

from scipy.linalg.cython_lapack cimport zheevr, zgeev
from scipy.linalg.cython_blas cimport zgemm, zgemv, zaxpy
from qutip.cy.spmath cimport (
    _zcsr_kron_core, _zcsr_kron, _zcsr_add, _zcsr_transpose, _zcsr_adjoint,
    _zcsr_mult, _safe_multiply,
)
from qutip.cy.spconvert cimport fdense2D_to_CSR
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.brtools cimport spec_func
from libc.math cimport fabs, fmin
from libc.float cimport DBL_MAX
from libcpp.vector cimport vector
from qutip.cy.sparse_structs cimport (CSR_Matrix, COO_Matrix)
from qutip.settings import eigh_unsafe
import numpy as np

include "sparse_routines.pxi"

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         real(double complex x)
    double cabs   "abs" (double complex x)
    double complex sqrt(double complex x)


cdef int use_zgeev = eigh_unsafe


@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex[::1,:] farray_alloc(int nrows):
    """
    Allocate a complex zero array in fortran-order for a
    square matrix.

    Parameters
    ----------
    nrows : int
        Number of rows and columns in the matrix.

    Returns
    -------
    fview : memview
        A zeroed memoryview in fortran-order.
    """
    cdef double complex * temp = <double complex *>PyDataMem_NEW_ZEROED(nrows*nrows,sizeof(complex))
    cdef complex[:,::1] cview = <double complex[:nrows, :nrows]> temp
    cdef complex[::1,:] fview = cview.T
    return fview


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void dense_add_mult(complex[::1,:] A,
                  complex[::1,:] B,
                  double complex alpha) nogil:
    """
    Performs the dense matrix multiplication A = A + (alpha*B)
    where A and B are complex 2D square matrices,
    and alpha is a complex coefficient.

    Parameters
    ----------
    A : ndarray
        Complex matrix in f-order that is to be overwritten
    B : ndarray
        Complex matrix in f-order.
    alpha : complex
        Coefficient in front of B.

    """
    cdef int nrows2 = A.shape[0]**2
    cdef int inc = 1
    zaxpy(&nrows2, &alpha, &B[0,0], &inc, &A[0,0], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZHEEVR(complex[::1,:] H, double * eigvals,
                    complex[::1,:] Z, int nrows):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.

    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigen values.
    Z : array_like
        Output array of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    if use_zgeev:
        ZGEEV(H, eigvals, Z, nrows)
        return
    cdef char jobz = b'V'
    cdef char rnge = b'A'
    cdef char uplo = b'L'
    cdef double vl=1, vu=1, abstol=0
    cdef int il=1, iu=1
    cdef int lwork = 18 * nrows
    cdef int lrwork = 24*nrows, liwork = 10*nrows
    cdef int info=0, M=0
    #These nee to be freed at end
    cdef int * isuppz = <int *>PyDataMem_NEW((2*nrows) * sizeof(int))
    cdef complex * work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((24*nrows) * sizeof(double))
    cdef int * iwork = <int *>PyDataMem_NEW((10*nrows) * sizeof(int))

    zheevr(&jobz, &rnge, &uplo, &nrows, &H[0,0], &nrows, &vl, &vu, &il, &iu, &abstol,
           &M, eigvals, &Z[0,0], &nrows, isuppz, work, &lwork,
          rwork, &lrwork, iwork, &liwork, &info)
    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(isuppz)
    PyDataMem_FREE(iwork)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZGEEV(complex[::1,:] H, double * eigvals,
                complex[::1,:] Z, int nrows):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.

    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigen values.
    Z : array_like
        Output array of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    cdef char jobvl = b'N'
    cdef char jobvr = b'V'
    cdef int i, j, k, lwork = -1
    cdef int same_eigv = 0
    cdef complex dot
    cdef complex wkopt
    cdef int info=0
    cdef complex * work

    #These need to be freed at end
    cdef complex * eival = <complex *>PyDataMem_NEW(nrows * sizeof(complex))
    cdef complex * vl = <complex *>PyDataMem_NEW(nrows * nrows *
                                                 sizeof(complex))
    cdef complex * vr = <complex *>PyDataMem_NEW(nrows * nrows *
                                                 sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((2*nrows) * sizeof(double))

    # First call to get lwork
    zgeev(&jobvl, &jobvr, &nrows, &H[0,0], &nrows,
          eival, vl, &nrows, vr, &nrows,
          &wkopt, &lwork, rwork, &info)
    lwork = int(real(wkopt))
    work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    # Solving
    zgeev(&jobvl, &jobvr, &nrows, &H[0,0], &nrows,
          eival, vl, &nrows, vr, &nrows, #&Z[0,0],
          work, &lwork, rwork, &info)
    for i in range(nrows):
        eigvals[i] = real(eival[i])
    # After using lapack, numpy...
    # lapack does not seems to have sorting function
    # zheevr sort but not zgeev
    cdef long[:] idx = np.argsort(np.array(<double[:nrows]> eigvals))
    for i in range(nrows):
        eigvals[i] = real(eival[idx[i]])
        for j in range(nrows):
            Z[j,i] = vr[j + idx[i]*nrows]

    for i in range(1, nrows):
        if cabs(eigvals[i] - eigvals[i-1]) < 1e-12:
            same_eigv += 1
            for j in range(same_eigv):
                dot = 0.
                for k in range(nrows):
                    dot += conj(Z[k,i-j-1]) * Z[k,i]
                for k in range(nrows):
                    Z[k,i] -= Z[k,i-j-1] * dot
                dot = 0.
                for k in range(nrows):
                    dot += conj(Z[k,i]) * Z[k,i]
                dot = sqrt(dot)
                for k in range(nrows):
                    Z[k,i] /= dot
        else:
            same_eigv = 0

    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(vl)
    PyDataMem_FREE(vr)
    PyDataMem_FREE(eival)
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
cdef void diag_liou_mult(double * diags, double complex * vec,
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
cdef complex[::1,:] dense_to_eigbasis(complex[::1,:] A, complex[::1,:] evecs,
                                    unsigned int nrows,
                                    double atol):
    cdef int kk
    cdef double complex * temp1 = ZGEMM(&A[0,0], &evecs[0,0],
                                       nrows, nrows, nrows, nrows, 0, 0)
    cdef double complex * eig_mat = ZGEMM(&evecs[0,0], temp1,
                                       nrows, nrows, nrows, nrows, 2, 0)
    PyDataMem_FREE(temp1)
    #Get view on ouput
    # Find all small elements and set to zero
    for kk in range(nrows**2):
        if cabs(eig_mat[kk]) < atol:
            eig_mat[kk] = 0
    cdef complex[:,::1] out = <complex[:nrows, :nrows]> eig_mat
    #This just gets the correct f-ordered view on the data
    cdef complex[::1,:] out_f = out.T
    return out_f


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex * vec_to_eigbasis(complex[::1] vec, complex[::1,:] evecs,
                                    unsigned int nrows):

    cdef size_t ii, jj
    cdef double complex * temp1 = ZGEMM(&vec[0], &evecs[0,0],
                                       nrows, nrows, nrows, nrows, 0, 0)
    cdef double complex * eig_vec = ZGEMM(&evecs[0,0], temp1,
                                       nrows, nrows, nrows, nrows, 2, 0)
    PyDataMem_FREE(temp1)
    return eig_vec


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[complex, ndim=1, mode='c'] vec_to_fockbasis(double complex * eig_vec,
                                                complex[::1,:] evecs,
                                                unsigned int nrows):

    cdef size_t ii, jj
    cdef np.npy_intp nrows2 = nrows**2
    cdef double complex * temp1 = ZGEMM(&eig_vec[0], &evecs[0,0],
                                       nrows, nrows, nrows, nrows, 0, 2)
    cdef double complex * fock_vec = ZGEMM(&evecs[0,0], temp1,
                                       nrows, nrows, nrows, nrows, 0, 0)
    PyDataMem_FREE(temp1)
    cdef np.ndarray[complex, ndim=1, mode='c'] out = \
                np.PyArray_SimpleNewFromData(1, &nrows2, np.NPY_COMPLEX128, fock_vec)
    PyArray_ENABLEFLAGS(out, np.NPY_ARRAY_OWNDATA)
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cop_super_term(complex[::1,:] cop, complex[::1,:] evecs,
                     double complex alpha, unsigned int nrows,
                     double atol):
    cdef size_t kk
    cdef CSR_Matrix mat1, mat2, mat3, mat4, mat5

    cdef complex[::1,:] cop_eig = dense_to_eigbasis(cop, evecs, nrows, atol)

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
    init_CSR(&mat2,
             _safe_multiply(mat1.nnz, mat1.nnz),
             _safe_multiply(mat1.nrows, mat1.nrows),
             _safe_multiply(mat1.ncols, mat1.ncols))
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



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cop_super_mult(complex[::1,:] cop, complex[::1,:] evecs,
                     double complex * vec,
                     double complex alpha,
                     double complex * out,
                     unsigned int nrows,
                     double atol) except *:
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
    init_CSR(
        &mat2,
        _safe_multiply(mat1.nnz, mat1.nnz),
        _safe_multiply(mat1.nrows, mat1.nrows),
        _safe_multiply(mat1.ncols, mat1.ncols),
    )
    _zcsr_kron_core(conj_data, mat1.indices, mat1.indptr,
                   mat1.data, mat1.indices, mat1.indptr,
                   &mat2,
                   mat1.nrows, mat1.nrows, mat1.ncols)

    #Do spmv with kron(cop.dag(), c)
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
    spmvpy(mat2.data,mat2.indices,mat2.indptr,
        vec, -0.5, &out[0], nrows**2)

    #Free mat1, mat2, and mat3
    free_CSR(&mat1)
    free_CSR(&mat2)
    free_CSR(&mat3)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void vec2mat_index(int nrows, int index, int[2] out) nogil:
    out[1] = index // nrows
    out[0] = index - nrows * out[1]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double skew_and_dwmin(double * evals, double[:,::1] skew,
                                unsigned int nrows) nogil:
    cdef double diff
    dw_min = DBL_MAX
    cdef size_t ii, jj
    for ii in range(nrows):
        for jj in range(nrows):
            diff = evals[ii] - evals[jj]
            skew[ii,jj] = diff
            if diff != 0:
                dw_min = fmin(fabs(diff), dw_min)
    return dw_min


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void br_term_mult(double t, complex[::1,:] A, complex[::1,:] evecs,
                double[:,::1] skew, double dw_min, spec_func spectral,
                double complex * vec, double complex * out,
                unsigned int nrows, int use_secular, double sec_cutoff,
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
    spmvpy(csr.data, csr.indices, csr.indptr, vec, 1, out, nrows**2)
    free_CSR(&csr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sparse_ZHEEVR(complex[::1,:] H, double * eigvals,
                    CSR_Matrix * evecs, int nrows, double atol):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.

    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigen values.
    evecs : CSR_Matrix
        Output csr matrix of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    cdef size_t jj, ii
    cdef char jobz = b'V'
    cdef char rnge = b'A'
    cdef char uplo = b'L'
    cdef double vl=1, vu=1, abstol=0
    cdef int il=1, iu=1
    cdef int lwork = 18 * nrows
    cdef int lrwork = 24*nrows, liwork = 10*nrows
    cdef int info=0, M=0
    #These nee to be freed at end
    cdef int * isuppz = <int *>PyDataMem_NEW((2*nrows) * sizeof(int))
    cdef complex * work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((24*nrows) * sizeof(double))
    cdef int * iwork = <int *>PyDataMem_NEW((10*nrows) * sizeof(int))
    cdef complex[:,::1] cZ = <complex[:nrows, :nrows]><complex *>PyDataMem_NEW(nrows**2 * sizeof(complex))
    cdef complex[::1,:] Z = cZ.T

    zheevr(&jobz, &rnge, &uplo, &nrows, &H[0,0], &nrows, &vl, &vu, &il, &iu, &abstol,
           &M, eigvals, &Z[0,0], &nrows, isuppz, work, &lwork,
          rwork, &lrwork, iwork, &liwork, &info)
    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(isuppz)
    PyDataMem_FREE(iwork)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")
    for jj in range(nrows):
        for ii in range(nrows):
            if cabs(Z[ii,jj]) < atol:
                Z[ii,jj] = 0
    fdense2D_to_CSR(Z, evecs, nrows, nrows)
    PyDataMem_FREE(&Z[0,0])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef CSR_Matrix sparse_to_eigbasis(complex[::1] Adata, int[::1] Aind, int[::1] Aptr,
                                CSR_Matrix * evecs, unsigned int nrows):
    cdef CSR_Matrix A, B, C, A_eig
    A.data = &Adata[0]
    A.indices = &Aind[0]
    A.indptr = &Aptr[0]
    A.nnz = A.indptr[nrows]
    A.nrows = nrows
    A.ncols = nrows
    A.max_length = A.nnz
    A.numpy_lock = 1
    A.is_set = 1

    _zcsr_mult(&A, evecs, &B)
    _zcsr_adjoint(evecs, &C)
    _zcsr_mult(&C, &B, &A_eig)
    sort_indices(&A_eig)
    free_CSR(&B)
    free_CSR(&C)
    return A_eig

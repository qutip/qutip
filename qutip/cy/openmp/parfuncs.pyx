#cython: language_level=3

import numpy as np
cimport numpy as cnp
cimport cython

cdef extern from "src/zspmv_openmp.hpp" nogil:
    void zspmvpy_openmp(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows, int nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_openmp(
        object super_op,
        complex[::1] vec,
        unsigned int nthr):
    """
    Sparse matrix, dense vector multiplication.
    Here the vector is assumed to have one-dimension.
    Matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    super_op : csr matrix
    vec : array
        Dense vector for multiplication.  Must be one-dimensional.

    Returns
    -------
    out : array
        Returns dense array.

    """
    return spmv_csr_openmp(super_op.data, super_op.indices, super_op.indptr, vec, nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_csr_openmp(complex[::1] data,
            int[::1] ind, int[::1] ptr, complex[::1] vec, unsigned int nthr):
    """
    Sparse matrix, dense vector multiplication.
    Here the vector is assumed to have one-dimension.
    Matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    vec : array
        Dense vector for multiplication.  Must be one-dimensional.

    Returns
    -------
    out : array
        Returns dense array.

    """
    cdef unsigned int num_rows = vec.shape[0]
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros(num_rows, dtype=complex)
    zspmvpy_openmp(&data[0], &ind[0], &ptr[0], &vec[0], 1.0, &out[0], num_rows, nthr)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void spmvpy_openmp(complex * data, int * ind, int * ptr,
            complex * vec,
            complex a,
            complex * out,
            unsigned int nrows,
            unsigned int nthr):

    zspmvpy_openmp(data, ind, ptr, vec, a, out, nrows, nthr)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs_openmp(
        double t,
        complex[::1] rho,
        complex[::1] data,
        int[::1] ind,
        int[::1] ptr,
        unsigned int nthr):

    cdef unsigned int nrows = rho.shape[0]
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = \
        np.zeros((nrows), dtype=complex)
    zspmvpy_openmp(&data[0], &ind[0], &ptr[0], &rho[0], 1.0, &out[0], nrows, nthr)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_openmp(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args,
        unsigned int nthr):

    H = H_func(t, args).data
    return -1j * spmv_csr_openmp(H.data, H.indices, H.indptr, psi, nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_with_state_openmp(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args,
        unsigned int nthr):

    H = H_func(t, psi, args)
    return -1j * spmv_csr_openmp(H.data, H.indices, H.indptr, psi, nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rho_func_td_openmp(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] rho,
        object L0,
        object L_func,
        object args,
        unsigned int nthr):
    cdef object L
    L = L0 + L_func(t, args).data
    return spmv_csr_openmp(L.data, L.indices, L.indptr, rho, nthr)

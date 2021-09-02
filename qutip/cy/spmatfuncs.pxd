#cython: language_level=3

cimport numpy as cnp
cimport cython
from libcpp cimport bool

include "parameters.pxi"

cpdef cnp.ndarray[CTYPE_t, ndim=1, mode="c"] spmv_csr(complex[::1] data,
                int[::1] ind, int[::1] ptr, complex[::1] vec)


cdef void spmvpy(complex * data,
                int * ind,
                int *  ptr,
                complex * vec,
                complex a,
                complex * out,
                unsigned int nrows)


cpdef cy_expect_rho_vec_csr(complex[::1] data,
                            int[::1] idx,
                            int[::1] ptr,
                            complex[::1] rho_vec,
                            int herm)


cpdef cy_expect_psi(object A,
                    complex[::1] vec,
                    bool isherm)


cpdef cy_expect_psi_csr(complex[::1] data,
                        int[::1] ind,
                        int[::1] ptr,
                        complex[::1] vec,
                        bool isherm)


cdef void _spmm_c_py(complex * data,
                     int * ind,
                     int * ptr,
                     complex * mat,
                     complex a,
                     complex * out,
                     unsigned int sp_rows,
                     unsigned int nrows,
                     unsigned int ncols)

cpdef void spmmpy_c(complex[::1] data,
                    int[::1] ind,
                    int[::1] ptr,
                    complex[:,::1] M,
                    complex a,
                    complex[:,::1] out)

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmmc(object sparse,
                                                   complex[:,::1] mat)

cdef void _spmm_f_py(complex * data,
                     int * ind,
                     int * ptr,
                     complex * mat,
                     complex a,
                     complex * out,
                     unsigned int sp_rows,
                     unsigned int nrows,
                     unsigned int ncols)

cpdef void spmmpy_f(complex[::1] data,
                    int[::1] ind,
                    int[::1] ptr,
                    complex[::1,:] mat,
                    complex a,
                    complex[::1,:] out)

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmmf(object sparse,
                                                   complex[::1,:] mat)

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmm(object sparse,
                                            cnp.ndarray[complex, ndim=2] mat)

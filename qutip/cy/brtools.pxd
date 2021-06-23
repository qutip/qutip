#cython: language_level=3

cimport numpy as np

#Spectral function with signature (w,t)
ctypedef complex (*spec_func)(double, double)

cdef complex[::1,:] farray_alloc(int nrows)

cpdef void dense_add_mult(complex[::1,:] A, complex[::1,:] B,
                  double complex alpha) nogil

cdef void ZHEEVR(complex[::1,:] H, double * eigvals,
                complex[::1,:] Z, int nrows)

cdef complex[::1,:] dense_to_eigbasis(complex[::1,:] A, complex[::1,:] evecs,
                                    unsigned int nrows, double atol)

cdef void diag_liou_mult(double * diags, double complex * vec,
                        double complex * out, unsigned int nrows) nogil

cdef double complex * vec_to_eigbasis(complex[::1] vec, complex[::1,:] evecs,
                                    unsigned int nrows)

cdef np.ndarray[complex, ndim=1, mode='c'] vec_to_fockbasis(double complex * eig_vec,
                                                complex[::1,:] evecs,
                                                unsigned int nrows)

cdef void cop_super_mult(complex[::1,:] cop, complex[::1,:] evecs,  double complex * vec,
                    double complex alpha,
                    double complex * out,
                    unsigned int nrows,
                    double atol) except *

cdef void vec2mat_index(int nrows, int index, int[2] out) nogil

cdef double skew_and_dwmin(double * evals, double[:,::1] skew,
                                unsigned int nrows) nogil


cdef void br_term_mult(double t, complex[::1,:] A, complex[::1,:] evecs,
                double[:,::1] skew, double dw_min, spec_func spectral,
                double complex * vec, double complex * out,
                unsigned int nrows, int use_secular, double sec_cutoff,
                double atol)

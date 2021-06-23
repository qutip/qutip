#cython: language_level=3

cimport numpy as cnp

from qutip.cy.brtools cimport spec_func


cdef void cop_super_mult_openmp(complex[::1,:] cop, complex[::1,:] evecs,  double complex * vec,
                    double complex alpha,
                    double complex * out,
                    unsigned int nrows,
                    unsigned int omp_thresh,
                    unsigned int nthr,
                    double atol) except *


cdef void br_term_mult_openmp(double t, complex[::1,:] A, complex[::1,:] evecs,
                double[:,::1] skew, double dw_min, spec_func spectral,
                double complex * vec, double complex * out,
                unsigned int nrows, int use_secular,
                double sec_cutoff,
                unsigned int omp_thresh,
                unsigned int nthr,
                double atol) except *

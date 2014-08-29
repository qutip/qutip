cimport numpy as np

# load complex functions from fortran
cdef extern double dznrm2_(int* N, complex* X, int* dx)

# load double functions from cblas
cdef extern from "cblas.h":
    double cblas_dnrm2(int N,double *X, int dx)
    double cblas_dznrm2(int N,complex *X, int dx)
    double cblas_ddot(int N, double *X, int dx, double *y, int dy)

# double 2-norms
def dnrm_1d(np.ndarray[double] x):
    cdef int n = len(x)
    cdef int dx = 1
    return cblas_dnrm2(n,<double*>x.data,dx)

def dnrm_2d(np.ndarray[double, ndim=2] x):
    cdef int n = len(x)
    cdef int dx = 1
    return cblas_dnrm2(n,<double*>x.data,dx)

#complex 2-norms
def znrm_1d(np.ndarray[complex] x):
    cdef int n = len(x)
    cdef int dx = 1
    return dznrm2_(&n,<complex*>x.data,&dx)

def znrm_2d(np.ndarray[complex, ndim=2] x):
    cdef int n = len(x)
    cdef int dx = 1
    return dznrm2_(&n,<complex*>x.data,&dx)

# double dot-product
def ddot_1d(np.ndarray[double] x, np.ndarray[double] y):
    cdef int n = len(x)
    cdef int dx = 1
    cdef int dy = 1
    return cblas_ddot(n,<double*>x.data,dx,<double*>y.data,dy)

def ddot_2d(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] y):
    cdef int n = len(x)
    cdef int dx = 1
    cdef int dy = 1
    return cblas_ddot(n,<double*>x.data,dx,<double*>y.data,dy)



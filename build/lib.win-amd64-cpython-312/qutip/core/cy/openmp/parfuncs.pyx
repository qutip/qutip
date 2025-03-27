#cython: language_level=3

cdef extern from "src/zspmv_openmp.hpp" nogil:
    void zspmvpy_openmp(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows, int nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void spmvpy_openmp(complex * data, int * ind, int * ptr,
                               complex * vec, complex a, complex * out,
                               unsigned int nrows, unsigned int nthr):
    zspmvpy_openmp(data, ind, ptr, vec, a, out, nrows, nthr)

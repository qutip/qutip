#cython: language_level=3

from qutip.cy.spmatfuncs cimport spmvpy as spmvpy_
from qutip.cy.spmatfuncs cimport _spmm_c_py, _spmm_f_py

IF HAVE_OPENMP:

    from qutip.cy.openmp.parfuncs cimport (spmvpy_openmp, spmmcpy_par,
                                           spmmfpy_omp)
    #from qutip.cy.openmp.dense_math cimport mvpy_openmp

    cdef void spmvpy(complex * data, int * ind, int *  ptr,
                     complex * vec, complex a, complex * out,
                     unsigned int nrows, unsigned int nthr):
        if nthr > 1:
            spmvpy_openmp(data, ind, ptr, vec, a, out, nrows, nthr)
        else:
            spmvpy_(data, ind, ptr, vec, a, out, nrows)

    cdef void spmmcpy(complex* data,
                           int* ind,
                           int* ptr,
                           complex* mat,
                           complex a,
                           complex* out,
                           int sp_rows,
                           unsigned int nrows,
                           unsigned int ncols,
                           int nthr):
        if nthr > 1:
            spmmcpy_par(data, ind, ptr, mat, a, out,
                        sp_rows, nrows, ncols, nthr)
        else:
            _spmm_c_py(data, ind, ptr, mat, a, out,
                       sp_rows, nrows, ncols)

    cdef void spmmfpy(complex* data, int* ind, int* ptr,
                           complex* mat,
                           complex a,
                           complex* out,
                           int sp_rows,
                           unsigned int nrows,
                           unsigned int ncols,
                           int nthr):
        if nthr > 1:
            spmmfpy_omp(data, ind, ptr, mat, a, out,
                        sp_rows, nrows, ncols, nthr)
        else:
            _spmm_f_py(data, ind, ptr, mat, a, out,
                       sp_rows, nrows, ncols)

    """
    cdef void mvpy(complex * data,
                   complex * vec, complex a, complex * out,
                   unsigned int nrows, unsigned int ncols, unsigned int nthr)

        mvpy_openmp(data, vec, a, out, nrows, ncols, nthr)
    """

    def check_omp():
        print("openmp detected")
ELSE:

    cdef void spmvpy(complex * data, int * ind, int *  ptr,
                     complex * vec, complex a, complex * out,
                     unsigned int nrows, unsigned int nthr):
        spmvpy_(data, ind, ptr, vec, a, out, nrows)

    cdef void spmmcpy(complex* data,
                           int* ind,
                           int* ptr,
                           complex* mat,
                           complex a,
                           complex* out,
                           int sp_rows,
                           unsigned int nrows,
                           unsigned int ncols,
                           int nthr):
        _spmm_c_py(data, ind, ptr, mat, a, out, sp_rows, nrows, ncols)

    cdef void spmmfpy(complex* data, int* ind, int* ptr,
                           complex* mat,
                           complex a,
                           complex* out,
                           int sp_rows,
                           unsigned int nrows,
                           unsigned int ncols,
                           int nthr):
        _spmm_f_py(data, ind, ptr, mat, a, out, sp_rows, nrows, ncols)

    """
    cdef void mvpy(complex * data,
                   complex * vec, complex a, complex * out,
                   unsigned int nrows, unsigned int ncols, unsigned int nthr)

        mvpy_openmp(complex * data,
                    complex * vec, complex a, complex * out,
                    unsigned int nrows, unsigned int ncols, unsigned int nthr)
    """

    def check_omp():
        print("openmp not-available")

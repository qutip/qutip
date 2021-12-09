#cython: language_level=3

cimport cython
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.openmp.parfuncs cimport spmvpy_openmp


@cython.boundscheck(False)
@cython.wraparound(False)
def _spmvpy(complex[::1] data,
            int[::1] ind,
            int[::1] ptr,
            complex[::1] vec,
            complex a,
            complex[::1] out):
    spmvpy(&data[0], &ind[0], &ptr[0], &vec[0], a, &out[0], vec.shape[0])



@cython.boundscheck(False)
@cython.wraparound(False)
def _spmvpy_openmp(complex[::1] data,
            int[::1] ind,
            int[::1] ptr,
            complex[::1] vec,
            complex a,
            complex[::1] out,
            unsigned int num_threads):
    spmvpy_openmp(&data[0], &ind[0], &ptr[0], &vec[0], a, &out[0], vec.shape[0], num_threads)

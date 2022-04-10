#cython: language_level=3
import warnings
import numpy as np
from qutip.settings import settings as qset
from qutip.core import Qobj

import sys

from libc.math cimport fabs
from libcpp cimport bool
from libcpp.vector cimport vector

cimport numpy as np
cimport cython

from qutip.core.data cimport CSR, idxint, csr
from qutip.core.data.add cimport add_csr
from qutip.solve._brtools cimport (
    vec2mat_index, dense_to_eigbasis, ZHEEVR, skew_and_dwmin,
    liou_from_diag_ham, cop_super_term
)

np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term(complex[::1,:] A, complex[::1,:] evecs, double[:,::1] skew,
                   double dw_min, object spectral, unsigned int nrows, int
                   use_secular, double sec_cutoff, double atol):

    cdef size_t kk
    cdef size_t I, J # vector index variables
    cdef int[2] ab, cd #matrix indexing variables
    cdef complex[::1,:] A_eig = dense_to_eigbasis(A, evecs, nrows, atol)
    cdef complex elem, ac_elem, bd_elem
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data
    cdef unsigned int nnz

    for I in range(nrows**2):
        vec2mat_index(nrows, I, ab)
        for J in range(nrows**2):
            vec2mat_index(nrows, J, cd)

            if (not use_secular) or (fabs(skew[ab[0],ab[1]]-skew[cd[0],cd[1]]) < (dw_min * sec_cutoff)):
                elem = (A_eig[ab[0],cd[0]]*A_eig[cd[1],ab[1]]) * 0.5
                elem *= (spectral(skew[cd[0],ab[0]])+spectral(skew[cd[1],ab[1]]))

                if (ab[0]==cd[0]):
                    ac_elem = 0
                    for kk in range(nrows):
                        ac_elem += A_eig[cd[1],kk]*A_eig[kk,ab[1]] * spectral(skew[cd[1],kk])
                    elem -= 0.5*ac_elem

                if (ab[1]==cd[1]):
                    bd_elem = 0
                    for kk in range(nrows):
                        bd_elem += A_eig[ab[0],kk]*A_eig[kk,cd[0]] * spectral(skew[cd[0],kk])
                    elem -= 0.5*bd_elem

                if (elem != 0):
                    coo_rows.push_back(I)
                    coo_cols.push_back(J)
                    coo_data.push_back(elem)

    PyDataMem_FREE(&A_eig[0,0])

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size())


@cython.boundscheck(False)
@cython.wraparound(False)
def bloch_redfield_tensor(object H, list a_ops, spectra_cb=None,
                 list c_ops=[], bool use_secular=True,
                 double sec_cutoff=0.1,
                 double atol = qset.core['atol']):
    """
    Calculates the time-independent Bloch-Redfield tensor for a system given
    a set of operators and corresponding spectral functions that describes the
    system's couplingto its environment.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        System Hamiltonian.

    a_ops : list
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra represented as Python
        functions.

    spectra_cb : list
        Depreciated.

    c_ops : list
        List of system collapse operators.

    use_secular : bool {True, False}
        Flag that indicates if the secular approximation should
        be used.

    sec_cutoff : float {0.1}
        Threshold for secular approximation.

    tol : float {qutip.settings.core['atol']}
       Threshold for removing small parameters.

    Returns
    -------

    R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        R is the Bloch-Redfield tensor and kets is a list eigenstates of the
        Hamiltonian.

    """
    cdef list _a_ops
    cdef object a, cop
    cdef CSR L
    cdef int K, kk
    cdef int nrows = H.shape[0]
    cdef list op_dims = H.dims
    cdef list sop_dims = [[op_dims[0], op_dims[0]], [op_dims[1], op_dims[1]]]
    cdef list ekets, ket_dims

    ket_dims = [op_dims[0], [1] * len(op_dims[0])]

    if not (spectra_cb is None):
        warnings.warn("The use of spectra_cb is depreciated.", DeprecationWarning)
        _a_ops = []
        for kk, a in enumerate(a_ops):
            _a_ops.append([a, spectra_cb[kk]])
        a_ops = _a_ops

    K = len(a_ops)

    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")

    for a in a_ops:
        if not isinstance(a[0], Qobj) or not a[0].isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    cdef complex[::1,:] H0 = H.full('F')
    cdef complex[::1,:] evecs = np.zeros((nrows,nrows), dtype=complex, order='F')
    cdef double[::1] evals = np.zeros(nrows, dtype=float)

    ZHEEVR(H0, &evals[0], evecs, nrows)
    L = liou_from_diag_ham(evals)

    for cop in c_ops:
        L = add_csr(L, cop_super_term(cop.full('F'), evecs, 1, nrows, atol))

    #only lindblad collapse terms
    if K == 0:
        ekets = [Qobj(np.asarray(evecs[:,k]), dims=ket_dims)
                 for k in range(nrows)]
        return Qobj(L, dims=sop_dims, type='super', copy=False), ekets

    #has some br operators and spectra
    cdef double[:,::1] skew = np.zeros((nrows,nrows), dtype=float)
    cdef double dw_min = skew_and_dwmin(&evals[0], skew, nrows)

    for a in a_ops:
        L = add_csr(L, _br_term(a[0].full('F'), evecs, skew, dw_min, a[1],
                                nrows, use_secular, sec_cutoff, atol))
    ekets = [Qobj(np.asarray(evecs[:,k]), dims=ket_dims) for k in range(nrows)]
    return Qobj(L, dims=sop_dims, type='super', copy=False), ekets

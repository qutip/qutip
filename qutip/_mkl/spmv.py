import numpy as np
from ctypes import POINTER, c_int, c_char, byref
from numpy.ctypeslib import ndpointer
import qutip.settings as qset
zcsrgemv = qset.mkl_lib.mkl_cspblas_zcsrgemv


def mkl_spmv(A, x):
    """
    sparse csr_spmv using MKL
    """
    m, _ = A.shape

    # Pointers to data of the matrix
    data = A.data.ctypes.data_as(ndpointer(np.complex128, ndim=1, flags='C'))
    indptr = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Allocate output, using same conventions as input
    if x.ndim == 1:
        y = np.empty(m, dtype=np.complex128, order='C')
    elif x.ndim == 2 and x.shape[1] == 1:
        y = np.empty((m, 1), dtype=np.complex128, order='C')
    else:
        raise Exception('Input vector must be 1D row or 2D column vector')

    # Now call MKL. This returns the answer in the last argument, which shares
    # memory with y.
    zcsrgemv(
        byref(c_char(bytes(b'N'))),
        byref(c_int(m)),
        data,
        indptr,
        indices,
        x.ctypes.data_as(ndpointer(np.complex128, ndim=1, flags='C')),
        y.ctypes.data_as(ndpointer(np.complex128, ndim=1, flags='C')),
    )
    return y

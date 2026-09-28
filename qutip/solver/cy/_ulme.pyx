from qutip.core.data cimport dense
from scipy.linalg cimport cython_blas as blas
cdef int _ONE=1

cpdef list split_dense(dense.Dense merged, int ncol):
    """
    Fast, little sanity checked matrix split for ULME internal loop
    """
    cdef int i, block_size = ncol * ncol
    cdef list out = []
    cdef dense.Dense tmp
    if merged.shape[0] != block_size:
        raise ValueError("Wrong shape")

    for i in range(merged.shape[1]):
        tmp = dense.Dense.__new__(dense.Dense)
        tmp.shape = (ncol, ncol)
        tmp._deallocate = False
        tmp._np = None
        tmp.data = &(merged.data[block_size * i])
        tmp.fortran = True
        out.append(tmp)
    return out

cpdef dense.Dense merge_dense(list datas):
    """
    Reverse of split_dense.
    Called much less, proper sanity checks.
    """
    if datas[0].shape[0] != datas[0].shape[1]:
        raise ValueError("Opers not square")

    cdef int i, block_size = datas[0].shape[0] * datas[0].shape[0]
    cdef dense.Dense tmp
    out = dense.zeros(block_size, len(datas), fortran=True)

    for i, tmp in enumerate(datas):
        if datas[0].shape[0] != tmp.shape[0] or datas[0].shape[0] != tmp.shape[1]:
            raise ValueError("Opers not all same shape")
        tmp =  tmp.reorder(fortran=True)
        blas.zcopy(&block_size, tmp.data, &_ONE, out.data + (block_size * i), &_ONE)
    return out

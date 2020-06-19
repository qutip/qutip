#cython: language_level=3

from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse.data import _data_matrix as scipy_data_matrix

from libc.string cimport memset
import numpy as np
cimport numpy as cnp
cimport cython

from . cimport base

cnp.import_array()

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void PyDataMem_FREE(void *ptr)


cdef _csr_matrix(data, indices, indptr, shape):
    """
    Factory method of scipy csr_matrix: we skip all the index type-checking
    because this takes tens of microseconds, and we already know we're in
    a sensible format.
    """
    cdef object out = scipy_csr_matrix.__new__(scipy_csr_matrix)
    # `_data_matrix` is the first object in the inheritance chain which
    # doesn't have a really slow __init__.
    scipy_data_matrix.__init__(out)
    out.data = data
    out.indices = indices
    out.indptr = indptr
    out._shape = shape
    return out


cdef class CSR(base.Data):
    """
    Data type for quantum objects storing its data in compressed sparse row
    (CSR) format.  This is similar to the `scipy` type
    `scipy.sparse.csr_matrix`, but significantly faster on many operations.
    You can retrieve a `scipy.sparse.csr_matrix` which views onto the same data
    using the `as_scipy()` method.
    """

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, arg=None, shape=None, copy=False):
        if not isinstance(arg, tuple) or len(arg) != 3:
            raise ValueError
        data = np.array(arg[0], dtype=np.complex128, copy=copy)
        col_index = np.array(arg[1], dtype=base.idxint_dtype, copy=copy)
        row_index = np.array(arg[2], dtype=base.idxint_dtype, copy=copy)
        self.data = data
        self.col_index = col_index
        self.row_index = row_index
        if shape is None:
            # row_index contains an extra element which is nnz.  We assume a
            # square matrix, like the original qutip.fast_csr_matrix, as
            # without a shape, the number of columns could be anything and
            # iterating to find the maximum column stored is slow.
            self.shape[0] = self.shape[1] = self.row_index.shape[0] - 1
        else:
            self.shape = shape
        self._scipy = _csr_matrix(data, col_index, row_index, self.shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def as_scipy(self):
        """
        Get a view onto this object as a `scipy.sparse.csr_matrix`.  The
        underlying data structures are exposed, such that modifications to the
        `data`, `indices` and `indptr` buffers in the resulting object will
        modify this object too.

        The arrays may be uninitialised.
        """
        # We store a reference to the scipy matrix not only for caching this
        # relatively expensive method, but also because we transferred
        # ownership of our data to the numpy arrays, and we can't allow them to
        # be collected while we're alive.
        if self._scipy is not None:
            return self._scipy
        data = cnp.PyArray_SimpleNewFromData(1, [self.data.size],
                                             cnp.NPY_COMPLEX128,
                                             &self.data[0])
        indices = cnp.PyArray_SimpleNewFromData(1, [self.col_index.size],
                                                base.idxint_DTYPE,
                                                &self.col_index[0])
        indptr = cnp.PyArray_SimpleNewFromData(1, [self.row_index.size],
                                               base.idxint_DTYPE,
                                               &self.row_index[0])
        PyArray_ENABLEFLAGS(data, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(indices, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(indptr, cnp.NPY_ARRAY_OWNDATA)
        self._scipy = _csr_matrix(data, indices, indptr, self.shape)
        return self._scipy

    def __dealloc__(self):
        # If we have a reference to a scipy type, then we've passed ownership
        # of the data to numpy, so we let it handle refcounting.
        if self._scipy is None:
            PyDataMem_FREE(&self.data[0])
            PyDataMem_FREE(&self.col_index[0])
            PyDataMem_FREE(&self.row_index[0])


cpdef CSR empty((base.idxint, base.idxint) shape, base.idxint size):
    """
    Allocate an empty CSR matrix of the given shape, with space for `size`
    elements in the `data` and `col_index` arrays.

    This does not initialise any of the memory returned, but sets the last
    element of `row_index` to 0 to indicate that there are 0 non-zero elements.
    """
    if size < 1:
        raise ValueError("size must be a positive integer.")
    cdef CSR out = CSR.__new__(CSR)
    cdef base.idxint row_size = shape[0] + 1
    out.shape = shape
    out.data =\
        <double complex [:size]> PyDataMem_NEW(size * sizeof(double complex))
    out.col_index =\
        <base.idxint [:size]> PyDataMem_NEW(size * sizeof(base.idxint))
    out.row_index =\
        <base.idxint [:row_size]> PyDataMem_NEW(row_size * sizeof(base.idxint))
    # Set the number of non-zero elements to 0.
    out.row_index[shape[0]] = 0
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR zeroes((base.idxint, base.idxint) shape):
    """
    Allocate the zero matrix with a given shape.  There will not be any room in
    the `data` and `col_index` buffers to add new elements.
    """
    # We always allocate matrices with at least one element to ensure that we
    # actually _are_ asking for memory (Python doesn't like allocating nothing)
    cdef CSR out = empty(shape, 1)
    out.data[0] = out.col_index[0] = 0
    memset(&out.row_index[0], 0, shape[0] + 1)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR identity(base.idxint dimension, double complex scale=1):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef CSR out = empty((dimension, dimension), size=dimension)
    cdef base.idxint i
    for i in range(dimension):
        out.data[i] = scale
        out.col_index[i] = i
        out.row_index[i] = i
    out.row_index[dimension] = dimension
    return out

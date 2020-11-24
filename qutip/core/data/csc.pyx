#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset, memcpy

cimport cython
import warnings

import numpy as np
cimport numpy as cnp
import scipy.sparse
from scipy.sparse import csc_matrix as scipy_csc_matrix
from scipy.sparse.data import _data_matrix as scipy_data_matrix

from qutip.core.data cimport base
from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data cimport csr
from qutip.core.data.base import idxint_dtype
from qutip.core.data.adjoint cimport (adjoint_csc, transpose_csc, conj_csc,
                                      transpose_csr)
from qutip.core.data.trace cimport trace_csc

cnp.import_array()

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)

__all__ = ['CSC']
cdef int _ONE = 1

cdef object _csc_matrix(data, indices, indptr, shape):
    """
    Factory method of scipy csc_matrix: we skip all the index type-checking
    because this takes tens of microseconds, and we already know we're in
    a sensible format.
    """
    cdef object out = scipy_csc_matrix.__new__(scipy_csc_matrix)
    # `_data_matrix` is the first object in the inheritance chain which
    # doesn't have a really slow __init__.
    scipy_data_matrix.__init__(out)
    out.data = data
    out.indices = indices
    out.indptr = indptr
    out._shape = shape
    return out


cdef class CSC(base.Data):
    """
    Data type for quantum objects storing its data in compressed sparse row
    (CSC) format.  This is similar to the `scipy` type
    `scipy.sparse.csc_matrix`, but significantly faster on many operations.
    You can retrieve a `scipy.sparse.csc_matrix` which views onto the same data
    using the `as_scipy()` method.
    """
    def __cinit__(self, *args, **kwargs):
        # By default, we want CSC to deallocate its memory (we depend on Cython
        # to ensure we don't deallocate a NULL pointer), and we only flip this
        # when we create a scipy backing.  Since creating the scipy backing
        # depends on knowing the shape, which happens _after_ data
        # initialisation and may throw an exception, it is better to have a
        # single flag that is set as soon as the pointers are assigned.
        self._deallocate = True

    def __init__(self, arg=None, shape=None, bint copy=False):
        # This is the Python __init__ method, so we do not care that it is not
        # super-fast C access.  Typically Cython code will not call this, but
        # will use a factory method in this module or at worst, call
        # CSC.__new__ and manually set everything.  We must be very careful in
        # this function that the deallocation is set up correctly if any
        # exceptions occur.
        cdef size_t ptr
        cdef base.idxint row
        cdef object data, col_index, row_index
        if isinstance(arg, scipy.sparse.spmatrix):
            arg = arg.tocsc()
            if shape is not None and shape != arg.shape:
                raise ValueError("".join([
                    "shapes do not match: ", str(shape), " and ", str(arg.shape),
                ]))
            shape = arg.shape
            arg = (arg.data, arg.indices, arg.indptr)
        if not isinstance(arg, tuple):
            raise TypeError("arg must be a scipy matrix or tuple")
        if len(arg) != 3:
            raise ValueError("arg must be a (data, row_index, col_index) tuple")
        data = np.array(arg[0], dtype=np.complex128, copy=copy, order='C')
        row_index = np.array(arg[1], dtype=idxint_dtype, copy=copy, order='C')
        col_index = np.array(arg[2], dtype=idxint_dtype, copy=copy, order='C')
        # This flag must be set at the same time as data, col_index and
        # row_index are assigned.  These assignments cannot raise an exception
        # in user code due to the above three lines, but further code may.
        self._deallocate = False
        self.data = <double complex *> cnp.PyArray_GETPTR1(data, 0)
        self.row_index = <base.idxint *> cnp.PyArray_GETPTR1(row_index, 0)
        self.col_index = <base.idxint *> cnp.PyArray_GETPTR1(col_index, 0)
        self.size = (cnp.PyArray_SIZE(data)
                     if cnp.PyArray_SIZE(data) < cnp.PyArray_SIZE(row_index)
                     else cnp.PyArray_SIZE(row_index))
        if shape is None:
            warnings.warn("instantiating CSC matrix of unknown shape")
            # row_index contains an extra element which is nnz.  We assume the
            # smallest matrix which can hold all these values by iterating
            # through the columns.  This is slow and probably inaccurate, since
            # there could be columns containing zero (hence the warning).
            self.shape[1] = cnp.PyArray_DIMS(col_index)[0] - 1
            row = 1
            for ptr in range(self.size):
                row = self.row_index[ptr] if self.row_index[ptr] > row else row
            self.shape[0] = row
        else:
            if not isinstance(shape, tuple):
                raise TypeError("shape must be a 2-tuple of positive ints")
            if not (len(shape) == 2
                    and isinstance(shape[0], int)
                    and isinstance(shape[1], int)
                    and shape[0] > 0
                    and shape[1] > 0):
                raise ValueError("shape must be a 2-tuple of positive ints")
            self.shape = shape
        # Store a reference to the backing scipy matrix so it doesn't get
        # deallocated before us.
        self._scipy = _csc_matrix(data, row_index, col_index, self.shape)

    def __reduce__(self):
        return (fast_from_scipy, (self.as_scipy(),))

    cpdef CSC copy(self):
        """
        Return a complete (deep) copy of this object.

        If the type currently has a scipy backing, such as that produced by
        `as_scipy`, this will not be copied.  The backing is a view onto our
        data, and a straight copy of this view would be incorrect.  We do not
        create a new view at copy time, since the user may only access this
        through a creation method, and creating it ahead of time would incur an
        unnecessary speed penalty for users who do not need it (including
        low-level C code).
        """
        cdef base.idxint nnz_ = nnz(self)
        cdef CSC out = empty_like(self)
        memcpy(out.data, self.data, nnz_*sizeof(double complex))
        memcpy(out.row_index, self.row_index, nnz_*sizeof(base.idxint))
        memcpy(out.col_index, self.col_index,
               (self.shape[1] + 1)*sizeof(base.idxint))
        return out

    cpdef object to_array(self):
        """
        Get a copy of this data as a full 2D, C-contiguous NumPy array.  This
        is not a view onto the data, and changes to new array will not affect
        the original data structure.
        """
        cdef cnp.npy_intp *dims = [self.shape[0], self.shape[1]]
        cdef object out = cnp.PyArray_ZEROS(2, dims, cnp.NPY_COMPLEX128, 0)
        cdef double complex [:, ::1] buffer = out
        cdef size_t col, ptr
        for col in range(self.shape[1]):
            for ptr in range(self.col_index[col], self.col_index[col + 1]):
                buffer[self.row_index[ptr], col] = self.data[ptr]
        return out

    cpdef object as_scipy(self, bint full=False):
        """
        Get a view onto this object as a `scipy.sparse.csc_matrix`.  The
        underlying data structures are exposed, such that modifications to the
        `data`, `indices` and `indptr` buffers in the resulting object will
        modify this object too.

        If `full` is False (the default), the output array is squeezed so that
        the SciPy array sees only the filled elements.  If `full` is True,
        SciPy will see the full underlying buffers, which may include
        uninitialised elements.  Setting `full=True` is intended for
        Python-space factory methods.  In all other use cases, `full=False` is
        much less error prone.
        """
        # We store a reference to the scipy matrix not only for caching this
        # relatively expensive method, but also because we transferred
        # ownership of our data to the numpy arrays, and we can't allow them to
        # be collected while we're alive.
        if self._scipy is not None:
            return self._scipy
        cdef cnp.npy_intp length = self.size if full else nnz(self)
        data = cnp.PyArray_SimpleNewFromData(1, [length],
                                             cnp.NPY_COMPLEX128,
                                             self.data)
        indices = cnp.PyArray_SimpleNewFromData(1, [length],
                                                base.idxint_DTYPE,
                                                self.row_index)
        indptr = cnp.PyArray_SimpleNewFromData(1, [self.shape[1] + 1],
                                               base.idxint_DTYPE,
                                               self.col_index)
        PyArray_ENABLEFLAGS(data, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(indices, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(indptr, cnp.NPY_ARRAY_OWNDATA)
        self._deallocate = False
        self._scipy = _csc_matrix(data, indices, indptr, self.shape)
        return self._scipy

    def __repr__(self):
        return "".join([
            "CSC(shape=", str(self.shape), ", nnz=", str(nnz(self)), ")",
        ])

    def __str__(self):
        return self.__repr__()

    @cython.initializedcheck(True)
    def __dealloc__(self):
        # If we have a reference to a scipy type, then we've passed ownership
        # of the data to numpy, so we let it handle refcounting and we don't
        # need to deallocate anything ourselves.
        if not self._deallocate:
            return
        if self.data != NULL:
            PyDataMem_FREE(self.data)
        if self.col_index != NULL:
            PyDataMem_FREE(self.col_index)
        if self.row_index != NULL:
            PyDataMem_FREE(self.row_index)

    cpdef CSC sort_indices(self):
        cdef CSR transposed = as_tr_csr(self, False)
        transposed.sort_indices()
        return self

    cpdef double complex trace(self):
        return trace_csc(self)

    cpdef CSC adjoint(self):
        return adjoint_csc(self)

    cpdef CSC conj(self):
        return conj_csc(self)

    cpdef CSC transpose(self):
        return transpose_csc(self)


cpdef CSC copy_structure(CSC matrix):
    """
    Return a copy of the input matrix with identical `col_index` and
    `row_index` matrices, and an allocated, but empty, `data`.  The returned
    pointers are all separately allocated, but contain the same information.

    This is intended for unary functions on CSC types that maintain the exact
    structure, but modify each non-zero data element without change their
    location.
    """
    cdef CSC out = empty_like(matrix)
    memcpy(out.row_index, matrix.row_index, nnz(matrix)*sizeof(base.idxint))
    memcpy(out.col_index, matrix.col_index, (matrix.shape[1] + 1)*sizeof(base.idxint))
    return out


cpdef inline base.idxint nnz(CSC matrix) nogil:
    """Get the number of non-zero elements of a CSC matrix."""
    return matrix.col_index[matrix.shape[1]]


cpdef CSC sorted(CSC matrix):
    cdef CSR transposed = as_tr_csr(matrix, False)
    cdef CSR tr_csr_sorted = csr.sorted(transposed)
    return from_tr_csr(tr_csr_sorted, False)


cpdef CSC empty(base.idxint rows, base.idxint cols, base.idxint size):
    """
    Allocate an empty CSC matrix of the given shape, with space for `size`
    elements in the `data` and `col_index` arrays.

    This does not initialise any of the memory returned, but sets the last
    element of `row_index` to 0 to indicate that there are 0 non-zero elements.
    """
    if size < 0:
        raise ValueError("size must be a positive integer.")
    # Python doesn't like allocating nothing.
    if size == 0:
        size += 1
    cdef CSC out = CSC.__new__(CSC)
    cdef base.idxint col_size = cols + 1
    out.shape = (rows, cols)
    out.size = size
    out.data =\
        <double complex *> PyDataMem_NEW(size * sizeof(double complex))
    out.row_index =\
        <base.idxint *> PyDataMem_NEW(size * sizeof(base.idxint))
    out.col_index =\
        <base.idxint *> PyDataMem_NEW(col_size * sizeof(base.idxint))
    # Set the number of non-zero elements to 0.
    out.col_index[cols] = 0
    return out


cpdef CSC empty_like(CSC other):
    return empty(other.shape[0], other.shape[1], nnz(other))


cpdef CSC zeros(base.idxint rows, base.idxint cols):
    """
    Allocate the zero matrix with a given shape.  There will not be any room in
    the `data` and `col_index` buffers to add new elements.
    """
    # We always allocate matrices with at least one element to ensure that we
    # actually _are_ asking for memory (Python doesn't like allocating nothing)
    cdef CSC out = empty(rows, cols, 1)
    out.data[0] = out.row_index[0] = 0
    memset(out.col_index, 0, (cols + 1) * sizeof(base.idxint))
    return out


cpdef CSC identity(base.idxint dimension, double complex scale=1):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef CSC out = empty(dimension, dimension, dimension)
    cdef base.idxint i
    for i in range(dimension):
        out.data[i] = scale
        out.row_index[i] = i
        out.col_index[i] = i
    out.col_index[dimension] = dimension
    return out


cpdef CSC fast_from_scipy(object sci):
    """
    Fast path construction from scipy.sparse.csc_matrix.  This does _no_ type
    checking on any of the inputs, and should consequently be considered very
    unsafe. This is primarily for use in the unpickling operation.
    """
    cdef CSC out = CSC.__new__(CSC)
    out.shape = sci.shape
    out._deallocate = False
    out._scipy = sci
    out.data = <double complex *> cnp.PyArray_GETPTR1(sci.data, 0)
    out.row_index = <base.idxint *> cnp.PyArray_GETPTR1(sci.indices, 0)
    out.col_index = <base.idxint *> cnp.PyArray_GETPTR1(sci.indptr, 0)
    out.size = cnp.PyArray_SIZE(sci.data)
    return out


cpdef CSC from_csr(CSR matrix):
    """Transform a CSR to CSC."""
    cdef CSR transposed = transpose_csr(matrix)
    return from_tr_csr(transposed, False)


cpdef CSC from_dense(Dense matrix):
    # Assume worst-case scenario for non-zero.
    cdef CSC out = empty(matrix.shape[0], matrix.shape[1],
                         matrix.shape[0]*matrix.shape[1])
    cdef size_t row, col, ptr_in, ptr_out=0, row_stride, col_stride
    row_stride = 1 if matrix.fortran else matrix.shape[1]
    col_stride = matrix.shape[0] if matrix.fortran else 1
    out.col_index[0] = 0
    for col in range(matrix.shape[1]):
        ptr_in = col_stride * col
        for row in range(matrix.shape[0]):
            if matrix.data[ptr_in] != 0:
                out.data[ptr_out] = matrix.data[ptr_in]
                out.row_index[ptr_out] = row
                ptr_out += 1
            ptr_in += row_stride
        out.col_index[col + 1] = ptr_out
    return out


cpdef CSR as_tr_csr(CSC matrix, bint copy=True):
    """ Return a CSR which is the transposed to this CSC.
    If copy is False, this CSR is a view on the CSC with no data ownership.
    """
    cdef CSR out = CSR.__new__(CSR)
    out.data = matrix.data
    out.col_index = matrix.row_index
    out.row_index = matrix.col_index
    out.size = matrix.size
    out._deallocate = False
    out.shape = (matrix.shape[1], matrix.shape[0])
    if copy:
        return out.copy()
    else:
        return out


cpdef CSC from_tr_csr(CSR matrix, bint copy=True):
    """ Return a CSC which is the transposed to this CSR.
    If copy is False, steal data ownership from the CSR.
    """
    cdef CSC out = CSC.__new__(CSC)
    out.data = matrix.data
    out.col_index = matrix.row_index
    out.row_index = matrix.col_index
    out.size = matrix.size
    out.shape = (matrix.shape[1], matrix.shape[0])
    if copy:
        out._deallocate = False
        return out.copy()
    else:
        out._deallocate = True
        matrix._deallocate = False
        return out


cpdef CSR to_csr(CSC matrix):
    cdef CSR transposed = as_tr_csr(matrix, False)
    return transpose_csr(transposed)


cpdef Dense to_dense(CSC matrix, bint fortran=False):
    cdef Dense out = Dense.__new__(Dense)
    out.shape = matrix.shape
    out.data = (
        <double complex *>
        PyDataMem_NEW_ZEROED(out.shape[0]*out.shape[1], sizeof(double complex))
    )
    out.fortran = fortran
    out._deallocate = True
    cdef size_t row, ptr_in, ptr_out, row_stride, col_stride, col
    row_stride = 1 if fortran else out.shape[1]
    col_stride = out.shape[0] if fortran else 1
    ptr_out = 0
    for col in range(out.shape[0]):
        for ptr_in in range(matrix.col_index[col], matrix.col_index[col + 1]):
            out.data[ptr_out + matrix.row_index[ptr_in]*row_stride] = matrix.data[ptr_in]
        ptr_out += col_stride
    return out

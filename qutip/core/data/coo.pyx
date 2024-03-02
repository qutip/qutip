#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memcpy

import warnings

import numpy as np
cimport numpy as cnp
import scipy.sparse
from scipy.sparse import coo_matrix as scipy_coo_matrix
try:
    from scipy.sparse.data import _data_matrix as scipy_data_matrix
except ImportError:
    # The file data was renamed to _data from scipy 1.8.0
    from scipy.sparse._data import _data_matrix as scipy_data_matrix

from qutip.core.data cimport base, csr, Dense
from qutip.core.data.adjoint cimport adjoint_coo, transpose_coo, conj_coo
from qutip.core.data.trace cimport trace_coo
from qutip.core.data.tidyup cimport tidyup_coo
from .base import idxint_dtype
from qutip.settings import settings

cnp.import_array()

cdef extern from *:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void *PyDataMem_NEW(size_t size)
    void PyDataMem_FREE(void *ptr)

# Very little should be exported on star-import, because most stuff will have
# naming collisions with other type modules.
__all__ = ['CSR']

cdef object _coo_matrix(data, row, col, shape):
    """
    Factory method of scipy csr_matrix: we skip all the index type-checking
    because this takes tens of microseconds, and we already know we're in
    a sensible format.
    """
    cdef object out = scipy_coo_matrix.__new__(scipy_coo_matrix)
    # `_data_matrix` is the first object in the inheritance chain which
    # doesn't have a really slow __init__.
    scipy_coo_matrix.__init__(out, (data , (row, col)), shape)
    return out


cdef class COO(base.Data):
    """
    Data type for quantum objects storing its data in coordinate
    (COO) format.  This is similar to the `scipy` type
    `scipy.sparse.coo_matrix`, but significantly faster on many operations.
    You can retrieve a `scipy.sparse.coo_matrix` which views onto the same data
    using the `as_scipy()` method.
    """
    def __cinit__(self, *args, **kwargs):
        # By default, we want COO to deallocate its memory (we depend on Cython
        # to ensure we don't deallocate a NULL pointer), and we only flip this
        # when we create a scipy backing.  Since creating the scipy backing
        # depends on knowing the shape, which happens _after_ data
        # initialisation and may throw an exception, it is better to have a
        # single flag that is set as soon as the pointers are assigned.
        self._deallocate = True

    def __init__(self, arg=None, shape=None, bint copy=True, bint tidyup=False):
        # This is the Python __init__ method, so we do not care that it is not
        # super-fast C access.  Typically Cython code will not call this, but
        # will use a factory method in this module or at worst, call
        # COO.__new__ and manually set everything.  We must be very careful in
        # this function that the deallocation is set up correctly if any
        # exceptions occur.
        cdef size_t ptr
        cdef base.idxint col, row
        cdef object data, col_index, row_index
        if isinstance(arg, scipy.sparse.spmatrix):
            arg = arg.tocoo()
            if shape is not None and shape != arg.shape:
                raise ValueError("".join([
                    "shapes do not match: ", str(shape), " and ", str(arg.shape),
                ]))
            shape = arg.shape
            arg = (arg.data, (arg.row, arg.col))
        if not isinstance(arg, tuple):
            raise TypeError("arg must be a scipy matrix or tuple")
        if len(arg) != 2 or not isinstance(arg[1], tuple) or len(arg[1]) != 2:
            raise ValueError("arg must be a (data, (row_index, col_index)) tuple")
        data = np.array(arg[0], dtype=np.complex128, copy=copy, order='C')
        row_index = np.array(arg[1][0], dtype=idxint_dtype, copy=copy, order='C')
        col_index = np.array(arg[1][1], dtype=idxint_dtype, copy=copy, order='C')
        # This flag must be set at the same time as data, col_index and
        # row_index are assigned.  These assignments cannot raise an exception
        # in user code due to the above three lines, but further code may.
        self._deallocate = False
        self.data = <double complex *> cnp.PyArray_GETPTR1(data, 0)
        self.col_index = <base.idxint *> cnp.PyArray_GETPTR1(col_index, 0)
        self.row_index = <base.idxint *> cnp.PyArray_GETPTR1(row_index, 0)
        self.size = cnp.PyArray_SIZE(data)
        self._nnz = <base.idxint> self.size
        if shape is None:
            warnings.warn("instantiating COO matrix of unknown shape")
            # row_index contains an extra element which is nnz.  We assume the
            # smallest matrix which can hold all these values by iterating
            # through the columns.  This is slow and probably inaccurate, since
            # there could be columns containing zero (hence the warning).
            for ptr in range(self.size):
                row = self.row_index[ptr] if self.row_index[ptr] > row else row
                col = self.col_index[ptr] if self.col_index[ptr] > col else col
            self.shape[0] = row
            self.shape[1] = col
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
        # deallocated before use.
        self._scipy = _coo_matrix(data, row_index, col_index, self.shape)
        if tidyup:
            tidyup_coo(self, settings.core['auto_tidyup_atol'], True)

    def __reduce__(self):
        return (fast_from_scipy, (self.as_scipy(),))

    cpdef COO copy(self):
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
        cdef COO out = empty_like(self)
        memcpy(out.data, self.data, nnz_*sizeof(double complex))
        memcpy(out.col_index, self.col_index, nnz_*sizeof(base.idxint))
        memcpy(out.row_index, self.row_index, nnz_*sizeof(base.idxint))
        out._nnz = nnz_
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
        cdef base.idxint ptr
        for ptr in range(self._nnz):
            buffer[self.row_index[ptr], self.col_index[ptr]] += self.data[ptr]
        return out

    cpdef object as_scipy(self, bint full=False):
        """
        Get a view onto this object as a `scipy.sparse.coo_matrix`.  The
        underlying data structures are exposed, such that modifications to the
        `data`, `indices` and `indptr` buffers in the resulting object will
        modify this object too.
        """
        # We store a reference to the scipy matrix not only for caching this
        # relatively expensive method, but also because we transferred
        # ownership of our data to the numpy arrays, and we can't allow them to
        # be collected while we're alive.
        if self._scipy is not None:
            return self._scipy
        cdef cnp.npy_intp length = self.size if full else self._nnz
        data = cnp.PyArray_SimpleNewFromData(1, [length],
                                             cnp.NPY_COMPLEX128,
                                             self.data)
        col = cnp.PyArray_SimpleNewFromData(1, [length],
                                                base.idxint_DTYPE,
                                                self.col_index)
        row = cnp.PyArray_SimpleNewFromData(1, [length],
                                               base.idxint_DTYPE,
                                               self.row_index)
        PyArray_ENABLEFLAGS(data, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(col, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(row, cnp.NPY_ARRAY_OWNDATA)
        self._deallocate = False
        self._scipy = _coo_matrix(data, row, col, self.shape)
        return self._scipy

    @property
    def nnz(self) -> int:
        return self._nnz

    cpdef double complex trace(self):
        return trace_coo(self)

    cpdef COO adjoint(self):
        return adjoint_coo(self)

    cpdef COO conj(self):
        return conj_coo(self)

    cpdef COO transpose(self):
        return transpose_coo(self)

    def __repr__(self):
        return "".join([
            "COO(shape=", str(self.shape), ", nnz=", str(nnz(self)), ")",
        ])

    def __str__(self):
        return self.__repr__()

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


cpdef COO fast_from_scipy(object sci):
    """
    Fast path construction from scipy.sparse.coo_matrix.  This does _no_ type
    checking on any of the inputs, and should consequently be considered very
    unsafe.  This is primarily for use in the unpickling operation.
    """
    cdef COO out = COO.__new__(COO)
    out.shape = sci.shape
    out._deallocate = False
    out._scipy = sci
    out.data = <double complex *> cnp.PyArray_GETPTR1(sci.data, 0)
    out.col_index = <base.idxint *> cnp.PyArray_GETPTR1(sci.col, 0)
    out.row_index = <base.idxint *> cnp.PyArray_GETPTR1(sci.row , 0)
    out.size = cnp.PyArray_SIZE(sci.data)
    out._nnz = out.size
    return out


cpdef COO copy_structure(COO matrix):
    """
    Return a copy of the input matrix with identical `col_index` and
    `row_index` matrices, and an allocated, but empty, `data`.  The returned
    pointers are all separately allocated, but contain the same information.

    This is intended for unary functions on COO types that maintain the exact
    structure, but modify each non-zero data element without change their
    location.
    """
    cdef COO out = empty_like(matrix)
    memcpy(out.col_index, matrix.col_index, nnz(matrix) * sizeof(base.idxint))
    memcpy(out.row_index, matrix.row_index, nnz(matrix) * sizeof(base.idxint))
    out._nnz = nnz(matrix)
    return out


cpdef inline base.idxint nnz(COO matrix) noexcept nogil:
    """Get the number of non-zero elements of a COO matrix."""
    return matrix._nnz

cpdef COO sorted(COO matrix):
    cdef COO out = empty_like(matrix)
    return out


cpdef COO empty(base.idxint rows, base.idxint cols, base.idxint size):
    """
    Allocate an empty COO matrix of the given shape, with space for `size`
    elements in the `data` and `col_index` arrays.

    This does not initialise any of the memory returned, but sets the last
    element of `row_index` to 0 to indicate that there are 0 non-zero elements.
    """
    if size < 0:
        raise ValueError("size must be a positive integer.")
    # Python doesn't like allocating nothing.
    if size == 0:
        size += 1
    cdef COO out = COO.__new__(COO)
    out.shape = (rows, cols)
    out.size = size
    out._nnz = 0
    out.data =\
        <double complex *> PyDataMem_NEW(size * sizeof(double complex))
    out.col_index =\
        <base.idxint *> PyDataMem_NEW(size * sizeof(base.idxint))
    out.row_index =\
        <base.idxint *> PyDataMem_NEW(size * sizeof(base.idxint))
    if not out.data: raise MemoryError()
    if not out.col_index: raise MemoryError()
    if not out.row_index: raise MemoryError()
    return out


cpdef COO empty_like(COO other):
    return empty(other.shape[0], other.shape[1], nnz(other))

cpdef COO expand(COO matrix, base.idxint size):
    cdef base.idxint nnz_ = nnz(matrix)
    if nnz_ > size:
        raise ValueError("size must be a greater than or equal to nnz")
    cdef COO out = empty(matrix.shape[0], matrix.shape[1], size)
    memcpy(out.data, matrix.data, nnz_*sizeof(double complex))
    memcpy(out.col_index, matrix.col_index, nnz_*sizeof(base.idxint))
    memcpy(out.row_index, matrix.row_index, nnz_*sizeof(base.idxint))
    out._nnz = nnz_
    return out

cpdef COO zeros(base.idxint rows, base.idxint cols):
    """
    Allocate the zero matrix with a given shape.  There will not be any room in
    the `data` and `col_index` buffers to add new elements.
    """
    # We always allocate matrices with at least one element to ensure that we
    # actually _are_ asking for memory (Python doesn't like allocating nothing)
    cdef COO out = empty(rows, cols, 0)
    return out


cpdef COO identity(base.idxint dimension, double complex scale=1):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef COO out = empty(dimension, dimension, dimension)
    cdef base.idxint i
    for i in range(dimension):
        out.data[i] = scale
        out.col_index[i] = i
        out.row_index[i] = i
    out._nnz = dimension
    return out

cpdef COO from_dense(Dense matrix):
    # Assume worst-case scenario for non-zero.
    cdef COO out = empty(matrix.shape[0], matrix.shape[1],
                         matrix.shape[0]*matrix.shape[1])
    cdef base.idxint row, col, ptr_in, ptr_out=0, row_stride, col_stride
    row_stride = 1 if matrix.fortran else matrix.shape[1]
    col_stride = matrix.shape[0] if matrix.fortran else 1
    for row in range(matrix.shape[0]):
        ptr_in = row_stride * row
        for col in range(matrix.shape[1]):
            if matrix.data[ptr_in] != 0:
                out.data[ptr_out] = matrix.data[ptr_in]
                out.col_index[ptr_out] = col
                out.row_index[ptr_out] = row
                ptr_out += 1
            ptr_in += col_stride
    out._nnz = ptr_out
    return out

cpdef COO from_csr(CSR matrix):
    cdef COO out = empty(matrix.shape[0], matrix.shape[1],
                         csr.nnz(matrix))
    cdef base.idxint row, ptr_out
    ptr_out = 0
    for row in range(matrix.shape[0]):
        for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
            out.data[ptr_out]  = matrix.data[ptr]
            out.row_index[ptr_out]  = row
            out.col_index[ptr_out]  = matrix.col_index[ptr]
            ptr_out += 1
    out._nnz = ptr_out
    return out


cdef inline base.idxint _diagonal_length(
    base.idxint offset, base.idxint n_rows, base.idxint n_cols,
) nogil:
    if offset > 0:
        return n_rows if offset <= n_cols - n_rows else n_cols - offset
    return n_cols if offset > n_cols - n_rows else n_rows + offset

cdef COO diag(
    double complex[:] diagonal, base.idxint offset,
    base.idxint n_rows, base.idxint n_cols,
):
    """
    Construct a COO matrix with a single non-zero diagonal.

    Parameters
    ----------
    diagonal : indexable of double complex
        The entries (including zeros) that should be placed on the diagonal in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal.

    offsets : idxint
        The index of the diagonals.  An offset of 0 is the main diagonal,
        positive values are above the main diagonal and negative ones are below
        the main diagonal.

    n_rows, n_cols : idxint
        The shape of the output.  The result does not need to be square, but
        the diagonal must be of the correct length to fit in.
    """
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape must be positive")
    cdef base.idxint nnz = len(diagonal)
    cdef base.idxint n_diag = _diagonal_length(offset, n_rows, n_cols)
    if nnz != n_diag:
        raise ValueError("incorrect number of diagonal elements")
    cdef COO out = empty(n_rows, n_cols, nnz)
    cdef base.idxint start_row = 0 if offset >= 0 else -offset
    cdef base.idxint col = 0 if offset <= 0 else offset
    cdef base.idxint ptr = 0
    for row in range(start_row, start_row + n_diag):
        out.col_index[ptr] = col
        out.row_index[ptr] = row
        out.data[ptr] = diagonal[ptr]
        col += 1
        ptr += 1
    out._nnz = ptr
    return out

#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset, memcpy

from libcpp cimport bool
from libcpp.algorithm cimport sort

cimport cython

from cpython cimport mem

import numbers
import warnings
import builtins
import numpy as np
cimport numpy as cnp
import scipy.sparse
from scipy.sparse import csr_matrix as scipy_csr_matrix
from functools import partial
from packaging.version import parse as parse_version
if parse_version(scipy.version.version) >= parse_version("1.14.0"):
    from scipy.sparse._data import _data_matrix as scipy_data_matrix
    # From scipy 1.14.0, a check that the input is not scalar was added for
    # sparse arrays.
    scipy_data_matrix = partial(scipy_data_matrix, arg1=(0,))
elif parse_version(scipy.version.version) >= parse_version("1.8.0"):
    # The file data was renamed to _data from scipy 1.8.0
    from scipy.sparse._data import _data_matrix as scipy_data_matrix
else:
    from scipy.sparse.data import _data_matrix as scipy_data_matrix
from scipy.linalg cimport cython_blas as blas

from qutip.core.data cimport base, Dense, Dia
from qutip.core.data.adjoint cimport adjoint_csr, transpose_csr, conj_csr
from qutip.core.data.trace cimport trace_csr
from qutip.core.data.tidyup cimport tidyup_csr
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

cdef int _ONE = 1

cdef object _csr_matrix(data, indices, indptr, shape):
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
    def __cinit__(self, *args, **kwargs):
        # By default, we want CSR to deallocate its memory (we depend on Cython
        # to ensure we don't deallocate a NULL pointer), and we only flip this
        # when we create a scipy backing.  Since creating the scipy backing
        # depends on knowing the shape, which happens _after_ data
        # initialisation and may throw an exception, it is better to have a
        # single flag that is set as soon as the pointers are assigned.
        self._deallocate = True

    def __init__(self, arg=None, shape=None, copy=True, bint tidyup=False):
        # This is the Python __init__ method, so we do not care that it is not
        # super-fast C access.  Typically Cython code will not call this, but
        # will use a factory method in this module or at worst, call
        # CSR.__new__ and manually set everything.  We must be very careful in
        # this function that the deallocation is set up correctly if any
        # exceptions occur.
        cdef size_t ptr
        cdef base.idxint col
        cdef object data, col_index, row_index
        if isinstance(arg, scipy.sparse.spmatrix):
            arg = arg.tocsr()
            if shape is not None and shape != arg.shape:
                raise ValueError("".join([
                    "shapes do not match: ", str(shape), " and ", str(arg.shape),
                ]))
            shape = arg.shape
            arg = (arg.data, arg.indices, arg.indptr)
        if not isinstance(arg, tuple):
            raise TypeError("arg must be a scipy matrix or tuple")
        if len(arg) != 3:
            raise ValueError("arg must be a (data, col_index, row_index) tuple")
        if np.lib.NumpyVersion(np.__version__) < '2.0.0b1':
            # np2 accept None which act as np1's False
            copy = builtins.bool(copy)
        data = np.array(arg[0], dtype=np.complex128, copy=copy, order='C')
        col_index = np.array(arg[1], dtype=idxint_dtype, copy=copy, order='C')
        row_index = np.array(arg[2], dtype=idxint_dtype, copy=copy, order='C')
        # This flag must be set at the same time as data, col_index and
        # row_index are assigned.  These assignments cannot raise an exception
        # in user code due to the above three lines, but further code may.
        self._deallocate = False
        self.data = <double complex *> cnp.PyArray_GETPTR1(data, 0)
        self.col_index = <base.idxint *> cnp.PyArray_GETPTR1(col_index, 0)
        self.row_index = <base.idxint *> cnp.PyArray_GETPTR1(row_index, 0)
        self.size = (cnp.PyArray_SIZE(data)
                     if cnp.PyArray_SIZE(data) < cnp.PyArray_SIZE(col_index)
                     else cnp.PyArray_SIZE(col_index))
        if shape is None:
            warnings.warn("instantiating CSR matrix of unknown shape")
            # row_index contains an extra element which is nnz.  We assume the
            # smallest matrix which can hold all these values by iterating
            # through the columns.  This is slow and probably inaccurate, since
            # there could be columns containing zero (hence the warning).
            self.shape[0] = cnp.PyArray_DIMS(row_index)[0] - 1
            col = 1
            for ptr in range(self.size):
                col = self.col_index[ptr] if self.col_index[ptr] > col else col
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
        # deallocated before us.
        self._scipy = _csr_matrix(data, col_index, row_index, self.shape)
        if tidyup:
            tidyup_csr(self, settings.core['auto_tidyup_atol'], True)

    def __reduce__(self):
        return (fast_from_scipy, (self.as_scipy(),))

    cpdef CSR copy(self):
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
        cdef CSR out = empty_like(self)
        memcpy(out.data, self.data, nnz_*sizeof(double complex))
        memcpy(out.col_index, self.col_index, nnz_*sizeof(base.idxint))
        memcpy(out.row_index, self.row_index,
               (self.shape[0] + 1)*sizeof(base.idxint))
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
        cdef size_t row, ptr
        for row in range(self.shape[0]):
            for ptr in range(self.row_index[row], self.row_index[row + 1]):
                buffer[row, self.col_index[ptr]] = self.data[ptr]
        return out

    cpdef object as_scipy(self, bint full=False):
        """
        Get a view onto this object as a `scipy.sparse.csr_matrix`.  The
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
                                                self.col_index)
        indptr = cnp.PyArray_SimpleNewFromData(1, [self.shape[0] + 1],
                                               base.idxint_DTYPE,
                                               self.row_index)
        PyArray_ENABLEFLAGS(data, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(indices, cnp.NPY_ARRAY_OWNDATA)
        PyArray_ENABLEFLAGS(indptr, cnp.NPY_ARRAY_OWNDATA)
        self._deallocate = False
        self._scipy = _csr_matrix(data, indices, indptr, self.shape)
        return self._scipy

    cpdef CSR sort_indices(self):
        cdef Sorter sort
        cdef base.idxint ptr
        cdef size_t row, diff, size=0
        for row in range(self.shape[0]):
            diff = self.row_index[row + 1] - self.row_index[row]
            size = diff if diff > size else size
        sort = Sorter(size)
        for row in range(self.shape[0]):
            ptr = self.row_index[row]
            diff = self.row_index[row + 1] - ptr
            sort.inplace(self, ptr, diff)
        return self

    cpdef double complex trace(self):
        return trace_csr(self)

    cpdef CSR adjoint(self):
        return adjoint_csr(self)

    cpdef CSR conj(self):
        return conj_csr(self)

    cpdef CSR transpose(self):
        return transpose_csr(self)

    def __repr__(self):
        return "".join([
            "CSR(shape=", str(self.shape), ", nnz=", str(nnz(self)), ")",
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


cpdef CSR fast_from_scipy(object sci):
    """
    Fast path construction from scipy.sparse.csr_matrix.  This does _no_ type
    checking on any of the inputs, and should consequently be considered very
    unsafe.  This is primarily for use in the unpickling operation.
    """
    cdef CSR out = CSR.__new__(CSR)
    out.shape = sci.shape
    out._deallocate = False
    out._scipy = sci
    out.data = <double complex *> cnp.PyArray_GETPTR1(sci.data, 0)
    out.col_index = <base.idxint *> cnp.PyArray_GETPTR1(sci.indices, 0)
    out.row_index = <base.idxint *> cnp.PyArray_GETPTR1(sci.indptr, 0)
    out.size = cnp.PyArray_SIZE(sci.data)
    return out


cpdef CSR copy_structure(CSR matrix):
    """
    Return a copy of the input matrix with identical `col_index` and
    `row_index` matrices, and an allocated, but empty, `data`.  The returned
    pointers are all separately allocated, but contain the same information.

    This is intended for unary functions on CSR types that maintain the exact
    structure, but modify each non-zero data element without change their
    location.
    """
    cdef CSR out = empty_like(matrix)
    memcpy(out.col_index, matrix.col_index, nnz(matrix) * sizeof(base.idxint))
    memcpy(out.row_index, matrix.row_index, (matrix.shape[0] + 1)*sizeof(base.idxint))
    return out


cpdef inline base.idxint nnz(CSR matrix) noexcept nogil:
    """Get the number of non-zero elements of a CSR matrix."""
    return matrix.row_index[matrix.shape[0]]


cdef bool _sorter_cmp_ptr(base.idxint *i, base.idxint *j) nogil:
    return i[0] < j[0]

cdef bool _sorter_cmp_struct(_data_col x, _data_col y) nogil:
    return x.col < y.col

ctypedef fused _swap_data:
    double complex
    base.idxint

cdef inline void _sorter_swap(_swap_data *a, _swap_data *b) noexcept nogil:
    a[0], b[0] = b[0], a[0]

cdef class Sorter:
    # Look on my works, ye mighty, and despair!
    #
    # This class has hard-coded sorts for up to three elements, for both
    # copying and in-place varieties.  Everything above that we delegate to a
    # proper sorting algorithm.
    def __init__(self, size_t size):
        self.size = size

    cdef void inplace(self, CSR matrix, base.idxint ptr, size_t size) nogil:
        cdef size_t n
        cdef base.idxint col0, col1, col2
        # Fast paths for tridiagonal matrices.  These fast paths minimise the
        # number of comparisons and swaps made.
        if size < 2:
            return
        if size == 2:
            if matrix.col_index[ptr] > matrix.col_index[ptr + 1]:
                _sorter_swap(matrix.col_index + ptr, matrix.col_index + ptr+1)
                _sorter_swap(matrix.data + ptr, matrix.data + ptr+1)
            return
        if size == 3:
            # Faster to store rather than re-dereference, and if someone
            # changes the data underneath us, we've got larger problems anyway.
            col0 = matrix.col_index[ptr]
            col1 = matrix.col_index[ptr + 1]
            col2 = matrix.col_index[ptr + 2]
            if col0 < col1:
                if col1 > col2:
                    _sorter_swap(matrix.col_index + ptr+1, matrix.col_index + ptr+2)
                    _sorter_swap(matrix.data + ptr+1, matrix.data + ptr+2)
                    if col0 > col2:
                        _sorter_swap(matrix.col_index + ptr, matrix.col_index + ptr+1)
                        _sorter_swap(matrix.data + ptr, matrix.data + ptr+1)
            elif col1 < col2:
                _sorter_swap(matrix.col_index + ptr, matrix.col_index + ptr+1)
                _sorter_swap(matrix.data + ptr, matrix.data + ptr+1)
                if col0 > col2:
                    _sorter_swap(matrix.col_index + ptr+1, matrix.col_index + ptr+2)
                    _sorter_swap(matrix.data + ptr+1, matrix.data + ptr+2)
            else:
                _sorter_swap(matrix.col_index + ptr, matrix.col_index + ptr+2)
                _sorter_swap(matrix.data + ptr, matrix.data + ptr+2)
            return
        # Now we actually have to do the sort properly.  It's easiest just to
        # copy the data into a temporary structure.
        if size > self.size or self.sort == NULL:
            # realloc(NULL, size) is equivalent to malloc(size), so there's no
            # problem if cols and argsort weren't allocated before.
            self.size = size if size > self.size else self.size
            with gil:
                self.sort = <_data_col *> mem.PyMem_Realloc(self.sort,
                                                            self.size * sizeof(_data_col))
        for n in range(size):
            self.sort[n].data = matrix.data[ptr + n]
            self.sort[n].col = matrix.col_index[ptr + n]
        sort(self.sort, self.sort + size, &_sorter_cmp_struct)
        for n in range(size):
            matrix.data[ptr + n] = self.sort[n].data
            matrix.col_index[ptr + n] = self.sort[n].col

    cdef void copy(self,
                   double complex *dest_data, base.idxint *dest_cols,
                   double complex *src_data, base.idxint *src_cols,
                   size_t size) nogil:
        cdef size_t n, ptr
        # Fast paths for small sizes.  Not pretty, but it speeds things up a
        # lot for up to triadiaongal systems (which are pretty common).
        if size == 0:
            return
        if size == 1:
            dest_cols[0] = src_cols[0]
            dest_data[0] = src_data[0]
            return
        if size == 2:
            if src_cols[0] < src_cols[1]:
                dest_cols[0] = src_cols[0]
                dest_data[0] = src_data[0]
                dest_cols[1] = src_cols[1]
                dest_data[1] = src_data[1]
            else:
                dest_cols[1] = src_cols[0]
                dest_data[1] = src_data[0]
                dest_cols[0] = src_cols[1]
                dest_data[0] = src_data[1]
            return
        if size == 3:
            if src_cols[0] < src_cols[1]:
                if src_cols[0] < src_cols[2]:
                    dest_cols[0] = src_cols[0]
                    dest_data[0] = src_data[0]
                    if src_cols[1] < src_cols[2]:
                        dest_cols[1] = src_cols[1]
                        dest_data[1] = src_data[1]
                        dest_cols[2] = src_cols[2]
                        dest_data[2] = src_data[2]
                    else:
                        dest_cols[2] = src_cols[1]
                        dest_data[2] = src_data[1]
                        dest_cols[1] = src_cols[2]
                        dest_data[1] = src_data[2]
                else:
                    dest_cols[1] = src_cols[0]
                    dest_data[1] = src_data[0]
                    dest_cols[2] = src_cols[1]
                    dest_data[2] = src_data[1]
                    dest_cols[0] = src_cols[2]
                    dest_data[0] = src_data[2]
            elif src_cols[0] < src_cols[2]:
                dest_cols[1] = src_cols[0]
                dest_data[1] = src_data[0]
                dest_cols[0] = src_cols[1]
                dest_data[0] = src_data[1]
                dest_cols[2] = src_cols[2]
                dest_data[2] = src_data[2]
            else:
                dest_cols[2] = src_cols[0]
                dest_data[2] = src_data[0]
                if src_cols[1] < src_cols[2]:
                    dest_cols[0] = src_cols[1]
                    dest_data[0] = src_data[1]
                    dest_cols[1] = src_cols[2]
                    dest_data[1] = src_data[2]
                else:
                    dest_cols[1] = src_cols[1]
                    dest_data[1] = src_data[1]
                    dest_cols[0] = src_cols[2]
                    dest_data[0] = src_data[2]
            return
        # Now we're left with the full case, and we have to sort properly.
        if size > self.size or self.argsort == NULL:
            # realloc(NULL, size) is equivalent to malloc(size), so there's no
            # problem if cols and argsort weren't allocated before.
            self.size = size if size > self.size else self.size
            with gil:
                self.argsort = (
                    <base.idxint **>
                    mem.PyMem_Realloc(self.argsort, self.size * sizeof(base.idxint *))
                )
        # We do the argsort with two levels of indirection to minimise memory
        # allocation and copying requirements when this function is being used
        # to assemble a CSR matrix under an operation which may change the
        # order of the columns (e.g. permute).  First the user makes the
        # columns accessible in some contiguous memory (or if they're not
        # changing them, they can just use the CSR buffers).  We put pointers
        # to each of those columns in the array which actually gets sorted
        # using a comparison function which dereferences the pointers and
        # compares the result.  After the sort, `argsort` will be the pointers
        # sorted according to the new column, and we know that the "lowest"
        # pointer in there has the value `src_cols`, so we can do pointer
        # arithmetic to know which element we should take.
        #
        # This is about 30-40% faster than allocating space for structs of
        # (double complex, idxint), copying in the data and column, sorting and
        # copying into the new arrays.  Allocating the structs actually
        # allocates more space than the pointer method (double complex is
        # very likely to be 2x the size of a pointer, _and_ the struct may need
        # extra padding to be aligned), so it's probably actually worse for
        # cache locality.  Despite the sort relying on pointer dereference in
        # this case, it's actually got very good cache locality.
        for n in range(size):
            self.argsort[n] = src_cols + n
        sort(self.argsort, self.argsort + size, &_sorter_cmp_ptr)
        for n in range(size):
            ptr = self.argsort[n] - src_cols
            dest_cols[n] = src_cols[ptr]
            dest_data[n] = src_data[ptr]

    def __dealloc__(self):
        if self.argsort != NULL:
            mem.PyMem_Free(self.argsort)
        if self.sort != NULL:
            mem.PyMem_Free(self.sort)

cpdef CSR sorted(CSR matrix):
    cdef CSR out = empty_like(matrix)
    cdef Sorter sort
    cdef base.idxint ptr
    cdef size_t row, diff, size=0
    memcpy(out.row_index, matrix.row_index, (matrix.shape[0] + 1) * sizeof(base.idxint))
    for row in range(matrix.shape[0]):
        diff = matrix.row_index[row + 1] - matrix.row_index[row]
        size = diff if diff > size else size
    sort = Sorter(size)
    for row in range(matrix.shape[0]):
        ptr = matrix.row_index[row]
        diff = matrix.row_index[row + 1] - ptr
        sort.copy(out.data + ptr, out.col_index + ptr,
                  matrix.data + ptr, matrix.col_index + ptr,
                  diff)
    return out


cpdef CSR empty(base.idxint rows, base.idxint cols, base.idxint size):
    """
    Allocate an empty CSR matrix of the given shape, with space for `size`
    elements in the `data` and `col_index` arrays.

    This does not initialise any of the memory returned, but sets the last
    element of `row_index` to 0 to indicate that there are 0 non-zero elements.
    """
    if size < 0:
        raise ValueError("size must be a positive integer.")
    # Python doesn't like allocating nothing.
    if size == 0:
        size += 1
    cdef CSR out = CSR.__new__(CSR)
    cdef base.idxint row_size = rows + 1
    out.shape = (rows, cols)
    out.size = size
    out.data =\
        <double complex *> PyDataMem_NEW(size * sizeof(double complex))
    out.col_index =\
        <base.idxint *> PyDataMem_NEW(size * sizeof(base.idxint))
    out.row_index =\
        <base.idxint *> PyDataMem_NEW(row_size * sizeof(base.idxint))
    if not out.data:
        raise MemoryError(
            f"Failed to allocate the `data` of a ({rows}, {cols}) "
            f"CSR array of {size} max elements."
        )
    if not out.col_index:
        raise MemoryError(
            f"Failed to allocate the `col_index` of a ({rows}, {cols}) "
            f"CSR array of {size} max elements."
        )
    if not out.row_index:
        raise MemoryError(
            f"Failed to allocate the `row_index` of a ({rows}, {cols}) "
            f"CSR array of {size} max elements."
        )
    # Set the number of non-zero elements to 0.
    out.row_index[rows] = 0
    return out


cpdef CSR empty_like(CSR other):
    return empty(other.shape[0], other.shape[1], nnz(other))


cpdef CSR zeros(base.idxint rows, base.idxint cols):
    """
    Allocate the zero matrix with a given shape.  There will not be any room in
    the `data` and `col_index` buffers to add new elements.
    """
    # We always allocate matrices with at least one element to ensure that we
    # actually _are_ asking for memory (Python doesn't like allocating nothing)
    cdef CSR out = empty(rows, cols, 1)
    out.data[0] = out.col_index[0] = 0
    memset(out.row_index, 0, (rows + 1) * sizeof(base.idxint))
    return out


cpdef CSR identity(base.idxint dimension, double complex scale=1):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    cdef CSR out = empty(dimension, dimension, dimension)
    cdef base.idxint i
    for i in range(dimension):
        out.data[i] = scale
        out.col_index[i] = i
        out.row_index[i] = i
    out.row_index[dimension] = dimension
    return out

cpdef CSR from_dense(Dense matrix):
    # Assume worst-case scenario for non-zero.
    cdef CSR out = empty(matrix.shape[0], matrix.shape[1],
                         matrix.shape[0]*matrix.shape[1])
    cdef size_t row, col, ptr_in, ptr_out=0, row_stride, col_stride
    row_stride = 1 if matrix.fortran else matrix.shape[1]
    col_stride = matrix.shape[0] if matrix.fortran else 1
    out.row_index[0] = 0
    for row in range(matrix.shape[0]):
        ptr_in = row_stride * row
        for col in range(matrix.shape[1]):
            if matrix.data[ptr_in] != 0:
                out.data[ptr_out] = matrix.data[ptr_in]
                out.col_index[ptr_out] = col
                ptr_out += 1
            ptr_in += col_stride
        out.row_index[row + 1] = ptr_out
    return out

cdef CSR from_coo_pointers(
    base.idxint *rows, base.idxint *cols, double complex *data,
    base.idxint n_rows, base.idxint n_cols, base.idxint nnz, double tol=0
):
    # Note that COO pointers may not be sorted in row-major order, and that
    # they may contain duplicate entries which should be implicitly summed.
    cdef Accumulator acc = acc_alloc(n_cols)
    cdef CSR out = empty(n_rows, n_cols, nnz)
    cdef double complex *data_tmp
    cdef base.idxint *cols_tmp
    cdef base.idxint row
    cdef size_t ptr_in, ptr_out, ptr_prev
    data_tmp = <double complex *> mem.PyMem_Malloc(nnz * sizeof(double complex))
    cols_tmp = <base.idxint *> mem.PyMem_Malloc(nnz * sizeof(base.idxint))
    if data_tmp == NULL or cols_tmp == NULL:
        raise MemoryError(
            f"Failed to allocate the memory needed for a ({n_rows}, {n_cols}) "
            f"CSR array with {nnz} elements."
        )
    with nogil:
        memset(out.row_index, 0, (n_rows + 1) * sizeof(base.idxint))
        for ptr_in in range(nnz):
            out.row_index[rows[ptr_in] + 1] += 1
        for ptr_out in range(n_rows):
            out.row_index[ptr_out + 1] += out.row_index[ptr_out]
        # out.row_index is currently in the normal output form, but we're
        # temporarily going to modify it to keep track of how many values we've
        # placed in each row as we iterate through.  At every state,
        # out.row_index[row] will contain a pointer to the next location that
        # an element should be placed in this row.
        for ptr_in in range(nnz):
            row = rows[ptr_in]
            ptr_out = out.row_index[row]
            cols_tmp[ptr_out] = cols[ptr_in]
            data_tmp[ptr_out] = data[ptr_in]
            out.row_index[row] += 1
        # Apply the scatter/gather pattern to find the actual number of
        # non-zero elements we're writing into each row (since there's a sum,
        # there may be some zeros of duplicates).  Remember we also need to
        # shift the row_index array back to what it was before as well.
        ptr_out = 0
        ptr_prev = 0
        for row in range(n_rows):
            for ptr_in in range(ptr_prev, out.row_index[row]):
                acc_scatter(&acc, data_tmp[ptr_in], cols_tmp[ptr_in])
            ptr_prev = out.row_index[row]
            out.row_index[row] = ptr_out
            ptr_out += acc_gather(&acc, out.data + ptr_out, out.col_index + ptr_out, tol)
            acc_reset(&acc)
        out.row_index[n_rows] = ptr_out
    mem.PyMem_Free(data_tmp)
    mem.PyMem_Free(cols_tmp)
    acc_free(&acc)
    return out


cpdef CSR from_dia(Dia matrix):
    cdef base.idxint col, diag, i, ptr=0
    cdef base.idxint nrows=matrix.shape[0], ncols=matrix.shape[1]
    cdef base.idxint nnz = matrix.num_diag * min(matrix.shape)
    cdef double complex[:] data = np.zeros(nnz, dtype=complex)
    cdef base.idxint[:] cols = np.zeros(nnz, dtype=idxint_dtype)
    cdef base.idxint[:] rows = np.zeros(nnz, dtype=idxint_dtype)

    for i in range(matrix.num_diag):
        diag = matrix.offsets[i]

        for col in range(ncols):
            if col - diag < 0 or col - diag >= nrows:
                continue
            data[ptr] = matrix.data[i * ncols + col]
            rows[ptr] = col - diag
            cols[ptr] = col
            ptr += 1

    return from_coo_pointers(&rows[0], &cols[0], &data[0], matrix.shape[0],
                             matrix.shape[1], nnz)


cdef inline base.idxint _diagonal_length(
    base.idxint offset, base.idxint n_rows, base.idxint n_cols,
) nogil:
    if offset > 0:
        return n_rows if offset <= n_cols - n_rows else n_cols - offset
    return n_cols if offset > n_cols - n_rows else n_rows + offset

cdef CSR diag(
    double complex[:] diagonal, base.idxint offset,
    base.idxint n_rows, base.idxint n_cols,
):
    """
    Construct a CSR matrix with a single non-zero diagonal.

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
    cdef CSR out = empty(n_rows, n_cols, nnz)
    cdef base.idxint start_row = 0 if offset >= 0 else -offset
    cdef base.idxint col = 0 if offset <= 0 else offset
    memset(out.row_index, 0, (start_row + 1) * sizeof(base.idxint))
    nnz = 0
    for row in range(start_row + 1, start_row + n_diag + 1):
        out.col_index[nnz] = col
        out.data[nnz] = diagonal[nnz]
        col += 1
        nnz += 1
        out.row_index[row] = nnz
    for row in range(start_row + n_diag + 1, n_rows + 1):
        out.row_index[row] = nnz
    return out

cdef CSR diags_(
    list diagonals, base.idxint[:] offsets,
    base.idxint n_rows, base.idxint n_cols,
):
    """
    Construct a CSR matrix from a list of diagonals and their offsets.  The
    offsets are assumed to be in sorted order.  This is the C-only interface to
    csr.diags, and inputs are not sanity checked (use the Python interface for
    that).

    Parameters
    ----------
    diagonals : list of indexable of double complex
        The entries (including zeros) that should be placed on the diagonals in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal (not checked).

    offsets : idxint[:]
        The indices of the diagonals.  These should be sorted and without
        duplicates.  `offsets[i]` is the location of the values `diagonals[i]`.
        An offset of 0 is the main diagonal, positive values are above the main
        diagonal and negative ones are below the main diagonal.

    n_rows, n_cols : idxint
        The shape of the output.  The result does not need to be square, but
        the diagonals must be of the correct length to fit in.
    """
    cdef size_t n_diagonals = len(diagonals)
    if n_diagonals == 0:
        return zeros(n_rows, n_cols)
    cdef base.idxint k, row, start_row, offset, nnz=0,
    cdef base.idxint min_k=n_diagonals, max_k=n_diagonals
    cdef double complex value
    for k in range(n_diagonals):
        offset = offsets[k]
        if offset >= 0 and min_k > k:
            min_k = k
        nnz += _diagonal_length(offset, n_rows, n_cols)
    cdef CSR out = empty(n_rows, n_cols, nnz)
    nnz = 0
    out.row_index[0] = 0
    for row in range(n_rows):
        if min_k > 0:
            offset = offsets[min_k - 1]
            start_row = 0 if offset >= 0 else -offset
            if start_row == row:
                min_k -= 1
        if max_k > 0:
            offset = offsets[max_k - 1]
            start_row = 0 if offset >= 0 else -offset
            if start_row + _diagonal_length(offset, n_rows, n_cols) - 1 < row:
                max_k -= 1
        for k in range(min_k, max_k):
            offset = offsets[k]
            value = diagonals[k][row if offset >= 0 else row + offset]
            if value == 0:
                continue
            out.data[nnz] = value
            out.col_index[nnz] = row + offset
            nnz += 1
        out.row_index[row + 1] = nnz
    return out

@cython.wraparound(True)
def diags(diagonals, offsets=None, shape=None):
    """
    Construct a CSR matrix from diagonals and their offsets.  Using this
    function in single-argument form produces a square matrix with the given
    values on the main diagonal.

    With lists of diagonals and offsets, the matrix will be the smallest
    possible square matrix if shape is not given, but in all cases the
    diagonals must fit exactly with no extra or missing elements.  Duplicated
    diagonals will be summed together in the output.

    Parameters
    ----------
    diagonals : sequence of array_like of complex or array_like of complex
        The entries (including zeros) that should be placed on the diagonals in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal and no more.

    offsets : sequence of integer or integer, optional
        The indices of the diagonals.  `offsets[i]` is the location of the
        values `diagonals[i]`.  An offset of 0 is the main diagonal, positive
        values are above the main diagonal and negative ones are below the main
        diagonal.

    shape : tuple, optional
        The shape of the output as (``rows``, ``columns``).  The result does
        not need to be square, but the diagonals must be of the correct length
        to fit in exactly.
    """
    cdef base.idxint n_rows, n_cols, offset
    try:
        diagonals = list(diagonals)
        if diagonals and np.isscalar(diagonals[0]):
            # Catch the case where we're being called as (for example)
            #   diags([1, 2, 3], 0)
            # with a single diagonal and offset.
            diagonals = [diagonals]
    except TypeError:
        raise TypeError("diagonals must be a list of arrays of complex") from None
    if offsets is None:
        if len(diagonals) == 0:
            offsets = []
        elif len(diagonals) == 1:
            offsets = [0]
        else:
            raise TypeError("offsets must be supplied if passing more than one diagonal")
    offsets = np.atleast_1d(offsets)
    if offsets.ndim > 1:
        raise ValueError("offsets must be a 1D array of integers")
    if len(diagonals) != len(offsets):
        raise ValueError("number of diagonals does not match number of offsets")
    if len(diagonals) == 0:
        if shape is None:
            raise ValueError("cannot construct matrix with no diagonals without a shape")
        else:
            n_rows, n_cols = shape
        return zeros(n_rows, n_cols)
    order = np.argsort(offsets)
    diagonals_ = []
    offsets_ = []
    prev, cur = None, None
    for i in order:
        cur = offsets[i]
        if cur == prev:
            diagonals_[-1] += np.asarray(diagonals[i], dtype=np.complex128)
        else:
            offsets_.append(cur)
            diagonals_.append(np.asarray(diagonals[i], dtype=np.complex128))
        prev = cur
    if shape is None:
        n_rows = n_cols = abs(offsets_[0]) + len(diagonals_[0])
    else:
        try:
            n_rows, n_cols = shape
        except (TypeError, ValueError):
            raise TypeError("shape must be a 2-tuple of positive integers")
        if n_rows < 0 or n_cols < 0:
            raise ValueError("shape must be a 2-tuple of positive integers")
    for i in range(len(diagonals_)):
        offset = offsets_[i]
        if len(diagonals_[i]) != _diagonal_length(offset, n_rows, n_cols):
            raise ValueError("given diagonals do not have the correct lengths")
    if n_rows == 0 and n_cols == 0:
        raise ValueError("can't produce a 0x0 matrix")
    if len(offsets) == 1:
        # Fast path for a single diagonal.
        return diag(diagonals_[0], offsets_[0], n_rows, n_cols)
    return diags_(
        diagonals_, np.array(offsets_, dtype=idxint_dtype), n_rows, n_cols,
    )


cpdef CSR _from_csr_blocks(
    base.idxint[:] block_rows, base.idxint[:] block_cols, CSR[:] block_ops,
    base.idxint n_blocks, base.idxint block_size
):
    """
    Construct a CSR from non-overlapping blocks.

    Each operator in ``block_ops`` should be a square CSR operator with
    shape ``(block_size, block_size)``. The output operator consists of
    ``n_blocks`` by ``n_blocks`` blocks and thus has shape
    ``(n_blocks * block_size, n_blocks * block_size)``.

    None of the operators should overlap (i.e. the list of block row and
    column pairs should be unique).

    Parameters
    ----------
    block_rows : sequence of base.idxint integers
        The block row for each operator. The block row should be in
        ``range(0, n_blocks)``.

    block_cols : sequence of base.idxint integers
        The block column for each operator. The block column should be in
        ``range(0, n_blocks)``.

    block_ops : sequence of CSR matrixes
        The operators corresponding to the rows and columns in ``block_rows``
        and ``block_cols``.

    n_blocks : base.idxint
        Number of blocks. The shape of the final matrix is
        (n_blocks * block, n_blocks * block).

    block_size : base.idxint
        Size of each block. The shape of matrices in ``block_ops`` is
        ``(block_size, block_size)``.
    """
    cdef CSR op
    cdef base.idxint shape = n_blocks * block_size
    cdef base.idxint nnz_ = 0
    cdef base.idxint n_ops = len(block_ops)
    cdef base.idxint i, j
    cdef base.idxint row_idx, col_idx

    # check arrays are the same length
    if len(block_rows) != n_ops or len(block_cols) != n_ops:
        raise ValueError(
            "The arrays block_rows, block_cols and block_ops should have"
            " the same length."
        )

    if n_ops == 0:
        return zeros(shape, shape)

    # check op shapes and calculate nnz
    for op in block_ops:
        nnz_ += nnz(op)
        if op.shape[0] != block_size or op.shape[1] != block_size:
            raise ValueError(
                "Block operators (block_ops) do not have the correct shape."
            )

    # check ops are ordered by (row, column)
    row_idx = block_rows[0]
    col_idx = block_cols[0]
    for i in range(1, n_ops):
        if (
            block_rows[i] < row_idx or
            (block_rows[i] == row_idx and block_cols[i] <= col_idx)
        ):
            raise ValueError(
                "The arrays block_rows and block_cols must be sorted"
                " by (row, column)."
            )
        row_idx = block_rows[i]
        col_idx = block_cols[i]

    if nnz_ == 0:
        return zeros(shape, shape)

    cdef CSR out = empty(shape, shape, nnz_)
    cdef base.idxint op_idx = 0
    cdef base.idxint prev_op_idx = 0
    cdef base.idxint end = 0
    cdef base.idxint row_pos, col_pos
    cdef base.idxint op_row, op_row_start, op_row_end, op_row_len

    out.row_index[0] = 0

    for row_idx in range(n_blocks):
        prev_op_idx = op_idx
        while op_idx < n_ops:
            if block_rows[op_idx] != row_idx:
                break
            op_idx += 1

        row_pos = row_idx * block_size
        for op_row in range(block_size):
            for i in range(prev_op_idx, op_idx):
                op = block_ops[i]
                if nnz(op) == 0:
                    # empty CSR matrices have uninitialized row_index entries.
                    # it's unclear whether users should ever see such matrixes
                    # but we support them here anyway.
                    continue
                col_idx = block_cols[i]
                col_pos = col_idx * block_size
                op_row_start = op.row_index[op_row]
                op_row_end = op.row_index[op_row + 1]
                op_row_len = op_row_end - op_row_start
                for j in range(op_row_len):
                    out.col_index[end + j] = op.col_index[op_row_start + j] + col_pos
                    out.data[end + j] = op.data[op_row_start + j]
                end += op_row_len
            out.row_index[row_pos + op_row + 1] = end

    return out

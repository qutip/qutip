#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False
#cython: cdivision=True

from qutip.core.data cimport Data, Dense, Dia, CSR

cdef extern from "<complex>" namespace "std" nogil:
    double complex _conj "conj"(double complex x)


cdef double complex NaN = float('NaN')


cdef class Data_iterator:
    """
    Iterator over Data instance.
    Data_iterator.next() return a (row, col, value) tuple.
    Data_iterator.nnz is the number of iteration to go over all values.

    There is no assurance on the order of the values.
    Sparse matrices' will only loop over stored values.

    ** After nnz call of next, it will return 0, 0, NaN **

    Parameters
    ----------
    Oper: Data
        Matrix to iterate over

    transpose: bool
        Whether to iterate over the transposed matrix.

    conj: bool
        Whether to return the conjugate of the
    """
    def __init__(self, Data oper, bint transpose, bint conj):
        self.nnz = 0
        self.transpose = transpose
        self.conj = conj

    cdef (int, int, double complex) next(self):
        return 0, 0, NaN


cdef class Dense_iterator(Data_iterator):
    def __init__(self, Dense oper, bint transpose, bint conj):
        self.position = -1
        self.oper = oper
        self.nnz = oper.shape[0] * oper.shape[1]
        self.transpose = transpose
        self.conj = conj

    cdef (int, int, double complex) next(self):
        self.position += 1

        if self.position >= self.nnz:
            return 0, 0, NaN

        cdef double complex val = self.oper.data[self.position]
        cdef int row, col
        if self.oper.fortran:
            row = self.position % self.oper.shape[0]
            col = self.position // self.oper.shape[0]
        else:
            row = self.position // self.oper.shape[1]
            col = self.position % self.oper.shape[1]

        if self.conj:
            val = _conj(val)
        if self.transpose:
            row, col = col, row
        return row, col, val


cdef class CSR_iterator(Data_iterator):
    def __init__(self, CSR oper, bint transpose, bint conj):
        self.row = 0
        self.idx = -1
        self.oper = oper
        self.nnz = oper.row_index[oper.shape[0]]
        self.transpose = transpose
        self.conj = conj

    cdef (int, int, double complex) next(self):
        self.idx += 1

        if self.idx >= self.nnz:
            return 0, 0, NaN

        cdef double complex val = self.oper.data[self.idx]
        cdef int col = self.oper.col_index[self.idx]

        while self.oper.row_index[self.row + 1] == self.idx:
            self.row += 1

        if self.conj:
            val = _conj(val)
        if self.transpose:
            return col, self.row, val
        else:
            return self.row, col, val


cdef class Dia_iterator(Data_iterator):
    def __init__(self, Dia oper, bint transpose, bint conj):
        self.oper = oper
        self.transpose = transpose
        self.conj = conj

        cdef int start, end

        self.nnz = 0
        for i in range(oper.num_diag):
            start = max(0, oper.offsets[i])
            end = min(oper.shape[1], oper.shape[0] + oper.offsets[i])
            self.nnz += max(end - start, 0)

        self.diag_N = -1
        self.diag_end = -1
        self.col = 0
        self.offset = 0

    cdef (int, int, double complex) next(self):
        self.col += 1

        while self.col >= self.diag_end:
            # While protect against badly formatted Dia.
            self.diag_N += 1
            if self.diag_N < self.oper.num_diag:
                self.offset = self.oper.offsets[self.diag_N]
                self.col = max(0, self.offset)
                self.diag_end = min(self.oper.shape[1], self.oper.shape[0] + self.offset)
            else:
                return 0, 0, NaN

        cdef double complex val = self.oper.data[self.diag_N * self.oper.shape[1] + self.col]
        if self.conj:
            val = _conj(val)

        if self.transpose:
            return (
                self.col,
                self.col - self.offset,
                val
            )

        return (
            self.col - self.offset,
            self.col,
            val
        )


cdef Data_iterator _make_iter(Data oper, bint transpose, bint conj):
    if type(oper) is Dense:
        iter = Dense_iterator(oper, transpose, conj)
    elif type(oper) is CSR:
        iter = CSR_iterator(oper, transpose, conj)
    elif type(oper) is Dia:
        iter = Dia_iterator(oper, transpose, conj)
    else:
        raise RuntimeError(f"Can't make an iter out of {type(oper)}")
    return iter

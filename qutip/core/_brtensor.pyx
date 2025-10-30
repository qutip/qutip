#cython: language_level=3
cimport cython

import qutip.core.data as _data
from qutip.core.data cimport Dense, CSR, Data, idxint, csr
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.cy._element cimport _BaseElement, _MapElement, _ProdElement
from qutip.core._brtools cimport _EigenBasisTransform
from qutip.core.qobj import Qobj

import numpy as np
import itertools
from libcpp.vector cimport vector
from libc.math cimport fabs, fmin

__all__ = []


cpdef enum TensorType:
    SPARSE = 0
    DENSE = 1
    DATA = 2


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data _br_term_data(Data A, double[:, ::1] spectrum,
                         double[:, ::1] skew, double cutoff):
    """
    Compute the contribution of A to the Bloch Redfield tensor.
    Computation are done using dispatched function.
    """
    cdef object cutoff_arr
    cdef int nrows = A.shape[0], a, b, c, d
    cdef Data S, I, AS, AST, out, C
    cdef type cls = type(A)

    S = _data.to(cls, _data.mul(_data.Dense(spectrum), 0.5))
    I = _data.identity[cls](nrows)
    AS = _data.multiply(A, S)
    AST = _data.multiply(A, _data.transpose(S))

    out = _data.kron(AST, _data.transpose(A))
    out = _data.add(out, _data.kron(A, _data.transpose(AS)))
    out = _data.sub(out, _data.kron(I, _data.transpose(_data.matmul(AS, A))))
    out = _data.sub(out, _data.kron(_data.matmul(A, AST), I))

    if cutoff == np.inf:
        return out

    # The cutoff_arr should be sparse most of the time, but it depend on the
    # cutoff and we cannot easily guess a nnz to make it efficiently...
    # But there is probably room from improvement.
    cutoff_arr = np.zeros((nrows*nrows, nrows*nrows), dtype=np.complex128)
    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        cutoff_arr[a * nrows + b, c * nrows + d] = 1.
    C = _data.to(cls, _data.Dense(cutoff_arr, copy=False))
    return _data.multiply(out, C)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Dense _br_term_dense(Data A, double[:, ::1] spectrum,
                           double[:, ::1] skew, double cutoff):
    """
    Compute the contribution of A to the Bloch Redfield tensor.
    Allocate a Dense array and fill it.
    """
    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
    cdef object np2term
    cdef Dense out
    cdef double complex[::1, :] out_array

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    out = _data.dense.zeros(nrows*nrows, nrows*nrows)
    out_array = out.as_ndarray()

    np2term = np.zeros((nrows, nrows, 2), dtype=np.complex128)
    ac_term = np2term[:, :, 0]
    bd_term = np2term[:, :, 1]


    for a in range(nrows):
        for b in range(nrows-1, -1, -1):
            if fabs(skew[a, b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    # TODO: we could use openmp to speed up.
    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        elem = A_mat[a, c] * A_mat[d, b] * 0.5
                        elem *= (spectrum[c, a] + spectrum[d, b])
                        if a == c:
                            elem = elem - 0.5 * ac_term[d, b]
                        if b == d:
                            elem = elem - 0.5 * bd_term[a, c]
                        out_array[a * nrows + b, c * nrows + d] = elem
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term_sparse(Data A, double[:, :] spectrum,
                          double[:, ::1] skew, double cutoff):
    """
    Compute the contribution of A to the Bloch Redfield tensor.
    Create it as coo pointers and return as CSR.
    """
    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k, d_min # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
    cdef double dskew
    cdef object np2term
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    np2term = np.zeros((nrows, nrows, 2), dtype=np.complex128)
    ac_term = np2term[:, :, 0]
    bd_term = np2term[:, :, 1]

    for a in range(nrows):
        for b in range(nrows-1, -1, -1):
            if fabs(skew[a, b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]
            elif skew[a, b] > cutoff:
                break

    # skew[a,b] = w[a] - w[b]
    # (w[a] - w[b] - w[c] + w[d]) < cutoff
    # w's are sorted so we can skip part of the loop.
    for a in range(nrows):
        for b in range(nrows):
            d_min = 0
            for c in range(nrows):
                if skew[a, b] - skew[c, nrows-1] <= -cutoff:
                    break
                for d in range(d_min, nrows):
                    dskew = skew[a, b] - skew[c, d]
                    if -dskew > cutoff:
                        d_min = d
                    elif fabs(dskew) < cutoff:
                        elem = (A_mat[a, c] * A_mat[d, b]) * 0.5
                        elem *= (spectrum[c, a] + spectrum[d, b])
                        if a == c:
                            elem -= 0.5 * ac_term[d, b]
                        if b == d:
                            elem -= 0.5 * bd_term[a, c]
                        if elem != 0:
                            coo_rows.push_back(a * nrows + b)
                            coo_cols.push_back(c * nrows + d)
                            coo_data.push_back(elem)
                    elif dskew >= cutoff:
                        break

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size()
    )


cdef class _BlochRedfieldElement(_BaseElement):
    """
    Element for individual Bloch Redfield collapse term.

    The tensor is computed in the ``eig_basis``, but is returned in the outside
    basis unless ``eig_basis`` is True. The ``matmul_data_t`` method also act
    on a state expected to be in that basis. It transform the state instead of
    the tensor as it is usually faster, the state being a density matrix in
    most cases.

    Diffenrent term can share a same instance of the eigen transform tool
    (``H`` as _EigenBasisTransform) so that the Hamiltonian is diagonalized
    only once even when needed by multiple terms.
    """
    cdef readonly _EigenBasisTransform H
    cdef readonly QobjEvo a_op
    cdef readonly Coefficient spectra
    cdef readonly double sec_cutoff
    cdef readonly size_t nrows
    cdef readonly (idxint, idxint) shape
    cdef readonly list dims
    cdef readonly object np_datas
    cdef readonly double[:, ::1] skew
    cdef readonly double[:, ::1] spectrum
    cdef readonly bint eig_basis
    cdef readonly TensorType tensortype

    def __init__(self, H, a_op, spectra, sec_cutoff, eig_basis=False,
                 dtype=None):
        if isinstance(H, _EigenBasisTransform):
            self.H = H
        else:
            self.H = _EigenBasisTransform(H)

        self.a_op = a_op
        self.spectra = spectra
        self.sec_cutoff = sec_cutoff
        self.eig_basis = eig_basis
        self.nrows = a_op.shape[0]
        self.dims = [a_op._dims, a_op._dims]

        dtype = dtype or ('dense' if sec_cutoff >= np.inf else 'sparse')
        self.tensortype = {
            'sparse': SPARSE,
            'dense': DENSE,
            'matrix': DATA,
            'data': DATA
        }[dtype]

        # Allocate some array
        # Let numpy manage memory
        self.np_datas = [np.zeros((self.nrows, self.nrows), dtype=np.float64),
                         np.zeros((self.nrows, self.nrows), dtype=np.float64)]
        self.skew = self.np_datas[0]
        self.spectrum = self.np_datas[1]

    cpdef double _compute_spectrum(self, double t) except *:
        "Compute the skew, spectrum and dw_min"
        cdef Coefficient spec
        cdef double dw_min = np.inf
        eigvals = self.H.eigenvalues(t)

        for col in range(0, self.nrows):
            self.skew[col, col] = 0.
            self.spectrum[col, col] = self.spectra(t, w=0).real
            for row in range(col, self.nrows):
                dw = eigvals[row] - eigvals[col]
                self.skew[row, col] = dw
                self.skew[col, row] = -dw
                if dw != 0:
                    dw_min = fmin(fabs(dw), dw_min)
                self.spectrum[row, col] = self.spectra(t, w=dw).real
                self.spectrum[col, row] = self.spectra(t, w=-dw).real
        return dw_min

    cdef Data _br_term(self, Data A_eig, double cutoff):
        if self.tensortype == DENSE:
            return _br_term_dense(A_eig, self.spectrum, self.skew, cutoff)
        elif self.tensortype == SPARSE:
            return _br_term_sparse(A_eig, self.spectrum, self.skew, cutoff)
        elif self.tensortype == DATA:
            return _br_term_data(A_eig, self.spectrum, self.skew, cutoff)
        raise ValueError('Invalid tensortype')

    cpdef object qobj(self, t):
        return Qobj(self.data(t), dims=self.dims, copy=False, superrep="super")

    cpdef object coeff(self, t):
        return 1.

    cpdef Data data(self, t):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        if self.eig_basis:
            return BR_eig
        return self.H.from_eigbasis(t, BR_eig)

    cdef Data matmul_data_t(self, t, Data state, Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        cdef Data A_eig, BR_eig

        if not self.eig_basis:
            state = self.H.to_eigbasis(t, state)
        if not self.eig_basis and out is not None:
            out = self.H.to_eigbasis(t, out)
        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        out = _data.add(_data.matmul(BR_eig, state, dtype=type(state)), out, dtype=type(state))
        if not self.eig_basis:
            out = self.H.from_eigbasis(t, out)
        return out

    def linear_map(self, f, anti=False):
        return _MapElement(self, [f])

    def replace_arguments(self, args, cache=None):
        if cache is None:
            return _BlochRedfieldElement(
                _EigenBasisTransform(QobjEvo(self.H.oper, args=args),
                                     type(self.H.oper) is CSR),
                QobjEvo(self.a_op, args=args),
                self.spectra,
                self.sec_cutoff
            )
        H = None
        for old, new in cache:
            if old is self:
                return new
            if old is self.H:
                H = new
        if H is None:
            H = _EigenBasisTransform(QobjEvo(self.H.oper, args=args),
                                     type(self.H.oper) is CSR)
        new = _BlochRedfieldElement(
            H, QobjEvo(self.a_op, args=args),
            self.spectra.replace_arguments(**args), self.sec_cutoff
        )
        cache.append((self, new))
        cache.append((self.H, H))
        return new

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    def __mul__(left, right):
        cdef _MapElement out
        if isinstance(left, _BlochRedfieldElement):
            out = _MapElement(left, [], right)
        if isinstance(right, _BlochRedfieldElement):
            out = _MapElement(right, [], left)
        return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data _br_cterm_data(Data A, Data B, double[:, ::1] spectrum,
                         double[:, ::1] skew, double cutoff):
    """
    Compute the contribution of A to the Bloch Redfield tensor.
    Computation are done using dispatched function.
    """
    cdef object cutoff_arr
    cdef int nrows = A.shape[0], a, b, c, d
    cdef Data S, I, P1, P2, P3, P4
    cdef type cls = type(A)

    S = _data.to(cls, _data.mul(_data.Dense(spectrum), 0.5))
    I = _data.identity[cls](nrows)

    P1 = _data.kron(_data.multiply(B, _data.transpose(S)), _data.transpose(A))
    P2 = _data.kron(B, _data.transpose(_data.multiply(A, S)))
    P3 = _data.kron(I, _data.transpose(_data.matmul(_data.multiply(A, S), B)))
    P4 = _data.kron(_data.matmul(A, _data.multiply(B, _data.transpose(S))), I)

    out = _data.add(_data.sub(P1, P3), _data.sub(P2, P4))

    if cutoff == np.inf:
        return out

    # The cutoff_arr should be sparse most of the time, but it depend on the
    # cutoff and we cannot easily guess a nnz to make it efficiently...
    # But there is probably room from improvement.
    cutoff_arr = np.zeros((nrows*nrows, nrows*nrows), dtype=np.complex128)
    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        cutoff_arr[a * nrows + b, c * nrows + d] = 1.
    C = _data.to(cls, _data.Dense(cutoff_arr, copy=False))
    return _data.multiply(out, C)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Dense _br_cterm_dense(Data A, Data B, double[:, ::1] spectrum,
                           double[:, ::1] skew, double cutoff):
    """
    Compute the contribution of A to the Bloch Redfield tensor.
    Allocate a Dense array and fill it.
    """
    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem
    cdef double complex[:,:] A_mat, B_mat, ac_term, bd_term
    cdef object np2term
    cdef Dense out
    cdef double complex[::1, :] out_array

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    if type(B) is Dense:
        B_mat = B.as_ndarray()
    else:
        B_mat = B.to_array()

    out = _data.dense.zeros(nrows*nrows, nrows*nrows)
    out_array = out.as_ndarray()

    np2term = np.zeros((nrows, nrows, 2), dtype=np.complex128)
    ac_term = np2term[:, :, 0]
    bd_term = np2term[:, :, 1]

    for a in range(nrows):
        for b in range(nrows-1, -1, -1):
            if fabs(skew[a, b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * B_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * B_mat[k, b] * spectrum[b, k]

    # TODO: we could use openmp to speed up.
    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        elem = B_mat[a, c] * A_mat[d, b] * spectrum[d, b]
                        elem += B_mat[a, c] * A_mat[d, b] * spectrum[c, a]
                        if a == c:
                            elem = elem - ac_term[d, b]
                        if b == d:
                            elem = elem - bd_term[a, c]
                        out_array[a * nrows + b, c * nrows + d] = elem * 0.5
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_cterm_sparse(Data A, Data B, double[:, :] spectrum,
                           double[:, ::1] skew, double cutoff):
    """
    Compute the contribution of A to the Bloch Redfield tensor.
    Create it as coo pointers and return as CSR.
    """
    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k, d_min # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
    cdef double dskew
    cdef object np2term
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    if type(B) is Dense:
        B_mat = B.as_ndarray()
    else:
        B_mat = B.to_array()

    np2term = np.zeros((nrows, nrows, 2), dtype=np.complex128)
    ac_term = np2term[:, :, 0]
    bd_term = np2term[:, :, 1]

    for a in range(nrows):
        for b in range(nrows-1, -1, -1):
            if fabs(skew[a, b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * B_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * B_mat[k, b] * spectrum[b, k]
            elif skew[a, b] > cutoff:
                break

    # skew[a,b] = w[a] - w[b]
    # (w[a] - w[b] - w[c] + w[d]) < cutoff
    # w's are sorted so we can skip part of the loop.
    for a in range(nrows):
        for b in range(nrows):
            d_min = 0
            for c in range(nrows):
                if skew[a, b] - skew[c, nrows-1] <= -cutoff:
                    break
                for d in range(d_min, nrows):
                    dskew = skew[a, b] - skew[c, d]
                    if -dskew > cutoff:
                        d_min = d
                    elif fabs(dskew) < cutoff:
                        elem = (B_mat[a, c] * A_mat[d, b]) * 0.5
                        elem *= (spectrum[c, a] + spectrum[d, b])
                        if a == c:
                            elem -= 0.5 * ac_term[d, b]
                        if b == d:
                            elem -= 0.5 * bd_term[a, c]
                        if elem != 0:
                            coo_rows.push_back(a * nrows + b)
                            coo_cols.push_back(c * nrows + d)
                            coo_data.push_back(elem)
                    elif dskew >= cutoff:
                        break

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size()
    )


cdef class _BlochRedfieldCrossElement(_BlochRedfieldElement):
    """
    Element for individual Bloch Redfield collapse cross-term.

    Single term in equation (6):
        A^a -> a_op
        A^b -> b_op
        spectra -> S_ab

    The tensor is computed in the ``eig_basis``, but is returned in the outside
    basis unless ``eig_basis`` is True. The ``matmul_data_t`` method also act
    on a state expected to be in that basis. It transform the state instead of
    the tensor as it is usually faster, the state being a density matrix in
    most cases.

    Different term can share a same instance of the eigen transform tool
    (``H`` as _EigenBasisTransform) so that the Hamiltonian is diagonalized
    only once even when needed by multiple terms.
    """
    cdef readonly QobjEvo b_op

    def __init__(self, H, a_op, b_op, spectra, sec_cutoff, eig_basis=False,
                 dtype=None):
        if isinstance(H, _EigenBasisTransform):
            self.H = H
        else:
            self.H = _EigenBasisTransform(H)

        self.a_op = a_op
        self.b_op = b_op
        self.spectra = spectra
        self.sec_cutoff = sec_cutoff
        self.eig_basis = eig_basis
        self.nrows = a_op.shape[0]
        self.dims = [a_op._dims, a_op._dims]

        dtype = dtype or ('dense' if sec_cutoff >= np.inf else 'sparse')
        self.tensortype = {
            'sparse': SPARSE,
            'dense': DENSE,
            'matrix': DATA,
            'data': DATA
        }[dtype]

        # Allocate some array
        # Let numpy manage memory
        self.np_datas = [np.zeros((self.nrows, self.nrows), dtype=np.float64),
                         np.zeros((self.nrows, self.nrows), dtype=np.float64)]
        self.skew = self.np_datas[0]
        self.spectrum = self.np_datas[1]

    cdef Data _br_cterm(self, Data A_eig, Data B_eig, double cutoff):
        if self.tensortype == DENSE:
            return _br_cterm_dense(A_eig, B_eig, self.spectrum, self.skew, cutoff)
        elif self.tensortype == DATA:
            return _br_cterm_data(A_eig, B_eig, self.spectrum, self.skew, cutoff)
        elif self.tensortype == SPARSE:
            return _br_cterm_sparse(A_eig, B_eig, self.spectrum, self.skew, cutoff)
        raise ValueError('Invalid tensortype')

    cpdef Data data(self, t):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        cdef Data A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        cdef Data B_eig = self.H.to_eigbasis(t, self.b_op._call(t))
        cdef Data BR_eig = self._br_cterm(A_eig, B_eig, cutoff)
        if self.eig_basis:
            return BR_eig
        return self.H.from_eigbasis(t, BR_eig)

    cdef Data matmul_data_t(self, t, Data state, Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        cdef Data A_eig, B_eig, BR_eig

        if not self.eig_basis:
            state = self.H.to_eigbasis(t, state)
        if not self.eig_basis and out is not None:
            out = self.H.to_eigbasis(t, out)
        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        B_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_cterm(A_eig, B_eig, cutoff)
        out = _data.add(_data.matmul(BR_eig, state), out)
        if not self.eig_basis:
            out = self.H.from_eigbasis(t, out)
        return out

    def replace_arguments(self, args, cache=None):
        if cache is None:
            return _BlochRedfieldCrossElement(
                _EigenBasisTransform(QobjEvo(self.H.oper, args=args),
                                     type(self.H.oper) is CSR),
                QobjEvo(self.a_op, args=args),
                QobjEvo(self.b_op, args=args),
                self.spectra,
                self.sec_cutoff
            )

        H = None
        for old, new in cache:
            if old is self:
                return new
            if old is self.H:
                H = new
        if H is None:
            H = _EigenBasisTransform(QobjEvo(self.H.oper, args=args),
                                     type(self.H.oper) is CSR)
        new = _BlochRedfieldElement(
            H, QobjEvo(self.a_op, args=args), QobjEvo(self.b_op, args=args),
            self.spectra.replace_arguments(**args), self.sec_cutoff
        )
        cache.append((self, new))
        cache.append((self.H, H))
        return new

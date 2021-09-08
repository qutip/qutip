#cython: language_level=3
cimport cython

import qutip.core.data as _data
from qutip.core.data cimport Dense, CSR, Data, idxint, csr
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.cy._element cimport _BaseElement, _MapElement, _ProdElement
from qutip.core._brtools cimport SpectraCoefficient, _EigenBasisTransform
from qutip import Qobj

import numpy as np
#from cython.parallel import prange
cimport openmp
from libcpp.vector cimport vector
from libc.float cimport DBL_MAX
from libc.math cimport fabs, fmin


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data _br_term_data(Data A, double[:, ::1] spectrum,
                         double[:, ::1] skew, double cutoff):
    # TODO:
    #    Working with Data would allow brmesolve to run on gpu etc.
    #    But it need point wise product.
    raise NotImplementedError


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Dense _br_term_dense(Data A, double[:, ::1] spectrum,
                           double[:, ::1] skew, double cutoff):

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
        for b in range(nrows):
            if fabs(skew[a, b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    for a in range(nrows): # prange(nrows, nogil=True, schedule='dynamic'):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    elem = 0.
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        elem = (A_mat[a, c] * A_mat[d, b] * 0.5 *
                                (spectrum[c, a] + spectrum[d, b]))

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

    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
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
        for b in range(nrows):
            if fabs(skew[a,b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a,b] - skew[c,d]) < cutoff:
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

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size()
    )


cdef class _BlochRedfieldElement(_BaseElement):
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

    cdef readonly Dense evecs, out, eig_vec, temp, op_eig

    def __init__(self, H, a_op, spectra, sec_cutoff, eig_basis=False):
        if isinstance(H, _EigenBasisTransform):
            self.H = H
        else:
            self.H = _EigenBasisTransform(H)
        self.a_op = a_op
        self.nrows = a_op.shape[0]
        self.shape = (self.nrows * self.nrows, self.nrows * self.nrows)
        self.dims = [a_op.dims, a_op.dims]

        self.spectra = spectra
        self.sec_cutoff = sec_cutoff
        self.eig_basis = eig_basis

        # Allocate some array
        # Let numpy manage memory
        self.np_datas = [np.zeros((self.nrows, self.nrows), dtype=np.float64),
                         np.zeros((self.nrows, self.nrows), dtype=np.float64)]
        self.skew = self.np_datas[0]
        self.spectrum = self.np_datas[1]

    cpdef double _compute_spectrum(self, double t) except *:
        cdef Coefficient spec
        cdef double dw_min = DBL_MAX
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
        if self.sec_cutoff >= DBL_MAX:
            return _br_term_dense(A_eig, self.spectrum, self.skew, cutoff)
        else:
            return _br_term_sparse(A_eig, self.spectrum, self.skew, cutoff)

    cpdef object qobj(self, double t):
        return Qobj(self.data(t), dims=self.dims, type="super",
                    copy=False, superrep="super")

    cpdef double complex coeff(self, double t) except *:
        return 1.

    cpdef Data data(self, double t):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        if self.eig_basis:
            return BR_eig
        return self.H.from_eigbasis(t, BR_eig)

    cdef Data matmul_data_t(self, double t, Data state, Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        cdef Data A_eig, BR_eig

        if not self.eig_basis:
            state = self.H.to_eigbasis(t, state)
        if not self.eig_basis and out is not None:
            out = self.H.to_eigbasis(t, out)
        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        out = _data.add(_data.matmul(BR_eig, state), out)
        if not self.eig_basis:
            out = self.H.from_eigbasis(t, out)
        return out

    def linear_map(self, f, anti=False):
        return _MapElement(self, [f])

    def replace_arguments(self, args, cache=None):
        if cache is None:
            return _BlochRedfieldElement(
                self.H,
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
            H, QobjEvo(self.a_op, args=args), self.spectra, self.sec_cutoff
        )
        cache.append((self, new))
        cache.append((self.H, H))
        return new

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    def __mul__(left, right):
        cdef _MapElement out
        if type(left) is _BlochRedfieldElement:
            out = _MapElement(left, [], right)
        if type(right) is _BlochRedfieldElement:
            out = _MapElement(right, [], left)
        return out

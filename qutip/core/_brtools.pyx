#cython: language_level=3
from libc.math cimport fabs, fmin
from libc.float cimport DBL_MAX

cimport numpy as cnp
import numpy as np
import warnings

cimport cython

from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data cimport Data, Dense, idxint
import qutip.core.data as _data
from qutip.core import Qobj
from scipy.linalg cimport cython_blas as blas


__all__ = ['SpectraCoefficient']


cdef class SpectraCoefficient(Coefficient):
    """
    Change a Coefficient with `t` dependence to one with `w` dependence to use
    in Bloch-Redfield tensor to allow array based coefficients to be used as
    spectral functions.
    If 2 coefficients are passed, the first one is the frequence response and
    the second is the time response.
    """
    def __init__(self, Coefficient coeff_w, Coefficient coeff_t=None, double w=0):
        self.coeff_t = coeff_t
        self.coeff_w = coeff_w
        self.w = w

    cdef complex _call(self, double t) except *:
        if self.coeff_t is None:
            return self.coeff_w(self.w)
        return self.coeff_t(t) * self.coeff_w(self.w)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return SpectraCoefficient(self.coeff_t, self.coeff_w, self.w)

    def replace_arguments(self, _args=None, *, w=None, **kwargs):
        if _args:
            kwargs.update(_args)
        if kwargs:
            return SpectraCoefficient(
                self.coeff_w.replace(**kwargs),
                self.coeff_t.replace(**kwargs) if self.coeff_t else None,
                kwargs.get('w', w or self.w)
              )
        if w is not None:
            return SpectraCoefficient(self.coeff_w, self.coeff_t, w)
        return self


cdef Data _apply_trans(Data original, int trans):
    """helper function for matmul_var_data, apply transform."""
    cdef Data out
    if trans == 0:
        out = original
    elif trans == 1:
        out = original.transpose()
    elif trans == 2:
        out = original.conj()
    elif trans == 3:
        out = original.adjoint()
    return out


cdef char _fetch_trans_code(int trans):
    """helper function for matmul_var_Dense, fetch the blas flag byte"""
    if trans == 0:
        return b'N'
    elif trans == 1:
        return b'T'
    elif trans == 3:
        return b'C'


cpdef Data matmul_var_data(Data left, Data right,
                           int transleft, int transright):
    """
    matmul which input matrices can be transposed or adjoint.
    out = transleft(left) @ transright(right)

    with trans[left, right]:
        0 : Normal
        1 : Transpose
        2 : Conjugate
        3 : Adjoint
    """
    # Should this be supported in data.matmul?
    # A.dag() @ A is quite common.
    cdef int size
    cdef double complex beta
    cdef char left_code, right_code
    if (
        type(left) is Dense
        and type(right) is Dense
        and left.shape[0] == left.shape[1]
        and left.shape[0] == right.shape[0]
        and left.shape[0] == right.shape[1]
    ):
        return matmul_var_Dense(left, right, transleft, transright)
    left = _apply_trans(left, transleft)
    right = _apply_trans(right, transright)
    return _data.matmul(left, right)


cdef Dense matmul_var_Dense(Dense left, Dense right,
                            int transleft, int transright):
    """
    matmul which input matrices can be transposed or adjoint.
    out = transleft(left) @ transright(right)

    with trans[left, right]:
       0 : Normal
       1 : Transpose
       2 : Conjugate
       3 : Adjoint
    """
    # blas support matmul for normal, transpose, adjoint for fortran ordered
    # matrices.
    if not (
        left.shape[0] == left.shape[1]
        and left.shape[0] == right.shape[0]
        and left.shape[0] == right.shape[1]
    ):
        raise ValueError("Only implemented for square operators")
    cdef Dense out, a, b
    cdef double complex alpha = 1., beta = 0.
    cdef int tleft, tright, size = left.shape[0]

    # Since fortran to C equivalent to transpose, fix the codes to fortran
    tleft = transleft ^ (not left.fortran)
    tright = transright ^ (not right.fortran)

    # blas.zgemm does not support adjoint.
    if tleft + tright == 5:
        # adjoint and conjugate, we can't transpose the output to use zgemm
        out = matmul_var_data(left, right, transleft-2, transright-2)
        return out.conj()
    if tleft == 2 or tright == 2:
        # Need a conjugate, we compute the transpose of the desired results.
        # A.conj @ B^op -> (B^T^op @ A.dag)^T
        out = _data.dense.empty(left.shape[0], right.shape[1], False)
        a, b = right, left
        tleft, tright = tright ^ 1, tleft ^ 1
    else:
        out = _data.dense.empty(left.shape[0], right.shape[1], True)
        a, b = left, right

    left_code = _fetch_trans_code(tleft)
    right_code = _fetch_trans_code(tright)
    blas.zgemm(&left_code, &right_code, &size, &size, &size,
               &alpha, (<Dense> a).data, &size, (<Dense> b).data, &size,
               &beta, (<Dense> out).data, &size)

    return out


class _eigen_qevo:
    """
    Callable function to represent the eigenvectors of a QobjEvo at a time
    ``t``.
    """
    def __init__(self, qevo):
        self.qevo = QobjEvo(qevo)  # Force a copy
        self.args = None
        # This is a base conversion operator, the eigen basis part of the dims
        # are flat.
        self.out_dims = [qevo.dims[0], [qevo.shape[1]]]

    def __call__(self, t, args):
        if args is not self.args:
            self.args = args
            self.qevo.arguments(self.args)
        _, data = _data.eigs(_data.to(Dense, self.qevo._call(t)), True, True)
        return Qobj(data, copy=False, dims=self.out_dims)


cdef class _EigenBasisTransform:
    """
    For an hermitian operator, compute the eigenvalues and eigenstates and do
    the base change to and from that eigenbasis.

    parameter
    ---------
    oper : QobjEvo
        Hermitian operator for which to compute the eigenbasis.

    sparse : bool [False]
        Deprecated
    """
    def __init__(self, QobjEvo oper, bint sparse=False):
        if oper.dims[0] != oper.dims[1]:
            raise ValueError
        if type(oper(0).data) in (_data.CSR, _data.Dia) and not sparse:
            oper = oper.to(Dense)
        elif type(oper(0).data) in (_data.CSR, _data.Dia) and sparse:
            warnings.warn(
                "Sparse Eigen solver is unstable and will be removed",
                DeprecationWarning
            )
        self.oper = oper
        self.isconstant = oper.isconstant
        self.size = oper.shape[0]

        if oper.isconstant:
            self._eigvals, self._evecs = _data.eigs(
                _data.to(Dense, self.oper._call(0)), True, True
            )
        else:
            self._evecs = None
            self._eigvals = None

        self._t = np.nan
        self._evecs_inv = None

    def as_Qobj(self):
        """Make an Qobj or QobjEvo of the eigenvectors."""
        if self.isconstant:
            return Qobj(self.evecs(0), dims=self.oper.dims)
        else:
            return QobjEvo(_eigen_qevo(self.oper))

    cdef void _compute_eigen(self, double t) except *:
        if self._t != t and not self.isconstant:
            self._t = t
            self._evecs_inv = None
            self._eigvals, self._evecs = _data.eigs(
                _data.to(Dense, self.oper._call(t)), True, True
            )

    cpdef object eigenvalues(self, double t):
        """
        Return the eigenvalues at ``t``.
        """
        self._compute_eigen(t)
        return self._eigvals

    cpdef Data evecs(self, double t):
        """
        Return the eigenstates at ``t``.
        """
        self._compute_eigen(t)
        return self._evecs

    cdef Data _inv(self, double t):
        if self._evecs_inv is None:
            self._evecs_inv = self.evecs(t).adjoint()
        return self._evecs_inv

    cdef Data _S_converter(self, double t):
        return _data.kron_transpose(self.evecs(t), self._inv(t))

    cdef Data _S_converter_inverse(self, double t):
        return _data.kron_transpose(self._inv(t), self.evecs(t))

    cpdef Data to_eigbasis(self, double t, Data fock):
        """
        Do the transformation of the :cls:`Qobj` ``fock`` to the basis where
        ``oper(t)`` is diagonalized.
        """
        # For Hermitian operator, the inverse of evecs is the adjoint matrix.
        # Blas include A.dag @ B in one operation. We use it if we can so we
        # don't make unneeded copy of evecs.
        cdef Data temp
        if fock.shape[0] == self.size and fock.shape[1] == 1:
            return matmul_var_data(self.evecs(t), fock, 3, 0)

        elif fock.shape[0] == self.size**2 and fock.shape[1] == 1:
            if type(fock) is Dense and (<Dense> fock).fortran:
                fock = _data.column_unstack_dense(fock, self.size, True)
                temp = _data.matmul(matmul_var_data(self.evecs(t), fock, 3, 0),
                                    self.evecs(t), dtype=type(fock))
                fock = _data.column_stack_dense(fock, True)
            else:
                fock = _data.column_unstack(fock, self.size, dtype=type(fock))
                temp = _data.matmul(matmul_var_data(self.evecs(t), fock, 3, 0),
                                    self.evecs(t), dtype=type(fock))
            if type(temp) is Dense:
                return _data.column_stack_dense(temp, True)
            return _data.column_stack(temp, dtype=type(fock))

        elif fock.shape[0] == self.size and fock.shape[0] == fock.shape[1]:
            return _data.matmul(matmul_var_data(self.evecs(t), fock, 3, 0),
                                self.evecs(t), dtype=type(fock))

        elif fock.shape[0] == self.size**2 and fock.shape[0] == fock.shape[1]:
            temp = self._S_converter_inverse(t)
            return _data.matmul(matmul_var_data(temp, fock, 3, 0), temp, dtype=type(fock))

        raise ValueError("Could not convert the Qobj's data to eigenbasis: "
                         "can't guess type from shape.")

    cpdef Data from_eigbasis(self, double t, Data eig):
        """
        Do the transformation of the :cls:`Qobj` ``eig`` in the basis where
        ``oper(t)`` is diagonalized to the outside basis.
        """
        cdef Data temp
        if eig.shape[0] == self.size and eig.shape[1] == 1:
            return _data.matmul(self.evecs(t), eig, dtype=type(eig))

        elif eig.shape[0] == self.size**2 and eig.shape[1] == 1:
            if type(eig) is Dense and (<Dense> eig).fortran:
                eig = _data.column_unstack_dense(eig, self.size, True)
                temp = matmul_var_data(_data.matmul(self.evecs(t), eig, dtype=type(eig)),
                                  self.evecs(t), 0, 3)
                eig = _data.column_stack_dense(eig, True)
            else:
                eig = _data.column_unstack(eig, self.size)
                temp = matmul_var_data(_data.matmul(self.evecs(t), eig, dtype=type(eig)),
                                  self.evecs(t), 0, 3)
            if type(temp) is Dense:
                return _data.column_stack_dense(temp, True)
            return _data.column_stack(temp, dtype=type(eig))

        elif eig.shape[0] == self.size and eig.shape[0] == eig.shape[1]:
            temp = self.evecs(t)
            return matmul_var_data(_data.matmul(temp, eig), temp, 0, 3)

        elif eig.shape[0] == self.size**2 and eig.shape[0] == eig.shape[1]:
            temp = self._S_converter_inverse(t)
            return _data.matmul(temp, matmul_var_data(eig, temp, 0, 3), dtype=type(eig))

        raise ValueError("Could not convert the Qobj's data from eigenbasis: "
                         "can't guess type from shape.")

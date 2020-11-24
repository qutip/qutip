#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True
# distutils: language = c++

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from libc.math cimport sqrt, NAN
from cpython.exc cimport PyErr_CheckSignals

cimport numpy as cnp
cnp.import_array()

from ..qobj import Qobj
from .. import data as _data

from qutip.core.data cimport CSR, Dense, CSC, Data
from qutip.core.data.add cimport add_csr, iadd_dense
from qutip.core.data.matmul cimport (matmul_csr, matmul_csr_dense_dense,
                                     matmul_dense, matmul_csc_dense_dense)
# TODO: handle dispatch properly.  We import rather than cimport because we
# have to call with Python semantics.
from qutip.core.data.expect cimport (
    expect_csr, expect_super_csr, expect_csr_dense, expect_super_csr_dense,
    expect_dense_dense, expect_super_dense_dense, expect_super_csc_dense,
    expect_csc_dense
)
from qutip.core.data.reshape cimport (column_stack_csr, column_stack_dense,
                                      column_unstack_dense)
from qutip.core.cy.coefficient cimport Coefficient


cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)


cdef Dense cy_matmul(Data left, Dense right, LTYPE layer_type):
    cdef Dense out
    if layer_type == CSR_TYPE:
        out = matmul_csr_dense_dense(left, right)
    elif layer_type == Dense_TYPE:
        out = matmul_dense(left, right)
    elif layer_type == CSC_TYPE:
        out = matmul_csc_dense_dense(left, right)
    else:
        out = _data.matmul(left, right)
    return out


cdef void cy_matmul_inplace(Data left, Dense right, LTYPE layer_type,
                            double complex scale, Dense out):
    if layer_type == CSR_TYPE:
        matmul_csr_dense_dense(left, right, scale, out)
    elif layer_type == Dense_TYPE:
        matmul_dense(left, right, scale, out)
    elif layer_type == CSC_TYPE:
        matmul_csc_dense_dense(left, right, scale, out)
    else:
        iadd_dense(out, _data.matmul[type(left), Dense, Dense](left, right, scale))


cdef double complex cy_expect(Data left, Dense right, LTYPE layer_type):
    cdef double complex out
    if layer_type == CSR_TYPE:
        out = expect_csr_dense(left, right)
    elif layer_type == Dense_TYPE:
        out = expect_dense_dense(left, right)
    elif layer_type == CSC_TYPE:
        out = expect_csc_dense(left, right)
    else:
        out = _data.expect(left, right)
    return out


cdef double complex cy_expect_super(Data left, Dense right, LTYPE layer_type):
    cdef double complex out
    if layer_type == CSR_TYPE:
        out = expect_super_csr_dense(left, right)
    elif layer_type == Dense_TYPE:
        out = expect_super_dense_dense(left, right)
    elif layer_type == CSC_TYPE:
        out = expect_super_csc_dense(left, right)
    else:
        out = _data.expect_super(left, right)
    return out


def count_types(data_obj, types):
    if isinstance(data_obj, CSR):
        types[0] += 1
    elif isinstance(data_obj, Dense):
        types[1] += 1
    elif isinstance(data_obj, CSC):
        types[2] += 1


def set_types(types):
    if types[1] == 0 and types[2] == 0:
        layer_type = CSR_TYPE
    elif types[0] == 0 and types[2] == 0:
        layer_type = Dense_TYPE
    elif types[1] == 0 and types[0] == 0:
        layer_type = CSC_TYPE
    else:
        layer_type = MIXED_TYPE
    return layer_type


cdef class CQobjEvo:
    """
    Data matmul(double t, Data matrix, Data out=None)
      Get the matrix multiplication of self with matrix and put the result in
      `out`, if supplied.  Always returns the object it stored the output in
      (even if it was passed).

    expect(double t, Data matrix)
      Get the expectation value at a time `t`.
      return expectation value, knows to use the super version or not.

    call(double t, double complex [:] coefficients=None, bint data=False)
      Get the full representation of this operator at time `t`.  If the
      coefficients are given, they are used instead and the underlying
      coefficient-getting functions are not called.  If `data` is True, then
      the data-layer object is returned instead of a full Qobj.
    """
    def __init__(self, constant, ops=None):
        cdef size_t i
        if not isinstance(constant, Qobj):
            raise TypeError("inputs must be Qobj")
        self.shape = constant.shape
        self.dims = constant.dims
        self.type = constant.type
        self.issuper = constant.issuper
        self.constant = constant.data
        self.n_ops = 0 if ops is None else len(ops)
        self.ops = [None] * self.n_ops
        self.coefficients = cnp.PyArray_EMPTY(1, [self.n_ops],
                                              cnp.NPY_COMPLEX128, False)
        self.coeff = [None] * self.n_ops
        types = [0, 0, 0]
        count_types(self.constant, types)
        for i in range(self.n_ops):
            vary = ops[i]
            qobj = vary.qobj
            if (
                qobj.shape != self.shape
                or qobj.type != self.type
                or qobj.dims != self.dims
            ):
                raise ValueError("not all inputs have the same structure")
            self.ops[i] = qobj.data
            self.coeff[i] = vary.coeff
            count_types(qobj.data, types)
        self.layer_type = set_types(types)

    def call(self, double t, object coefficients=None, bint data=False):
        cdef Data out = self.constant.copy()
        cdef Py_ssize_t i
        if coefficients is None:
            self._factor(t)
            coefficients = self.coefficients
        elif len(coefficients) != self.n_ops:
            raise TypeError(
                "got " + str(len(coefficients)) + " coefficients,"
                + " but expected " + str(self.n_ops)
            )
        for i in range(len(self.ops)):
            out = _data.add(out, <Data> self.ops[i], scale=coefficients[i])
        if data:
            return out
        return Qobj(out, dims=self.dims, type=self.type,
                    superrep=self.superrep, copy=False)

    cdef void _factor(self, double t) except *:
        cdef size_t i
        cdef Coefficient coeff
        for i in range(self.n_ops):
            coeff = <Coefficient> self.coeff[i]
            self.coefficients[i] = coeff._call(t)
        return

    cpdef Data matmul(self, double t, Data matrix):
        cdef size_t i
        if isinstance(matrix, Dense):
            return self.matmul_dense(t, matrix)
        self._factor(t)
        out = _data.matmul(self.constant, matrix)
        for i in range(self.n_ops):
            out = _data.add(out,
                            _data.matmul(<Data> self.ops[i],
                                         matrix,
                                         self.coefficients[i])
                )
        return out

    cpdef Dense matmul_dense(self, double t, Dense matrix, Dense out=None):
        cdef size_t i
        self._factor(t)
        if out is None:
            out = cy_matmul(self.constant, matrix, self.layer_type)
        else:
            cy_matmul_inplace(self.constant, matrix, self.layer_type, 1, out)
        for i in range(self.n_ops):
            cy_matmul_inplace(<Data> self.ops[i], matrix, self.layer_type,
                                    self.coefficients[i], out)
        return out

    cpdef double complex expect(self, double t, Data matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        cdef size_t i
        cdef double complex out
        self._factor(t)
        if self.issuper:
            if matrix.shape[1] != 1:
                matrix = _data.column_stack(matrix)
            out = _data.expect_super(self.constant, matrix)
            for i in range(self.n_ops):
                out += self.coefficients[i] * _data.expect_super(self.ops[i], matrix)
        else:
            out = _data.expect(self.constant, matrix)
            for i in range(self.n_ops):
                out += self.coefficients[i] * _data.expect(self.ops[i], matrix)
        return out

    cpdef double complex expect_dense(self, double t, Dense matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        cdef size_t nrow = matrix.shape[0]
        cdef size_t i
        cdef double complex out
        self._factor(t)
        if self.issuper:
            if matrix.shape[1] != 1:
                matrix = column_stack_dense(matrix, inplace=matrix.fortran)
            out = cy_expect_super(self.constant, matrix, self.layer_type)
            for i in range(self.n_ops):
                out += self.coefficients[i] * cy_expect_super(<Data> self.ops[i], matrix, self.layer_type)
            if matrix.fortran:
                column_unstack_dense(matrix, nrow, inplace=matrix.fortran)
        else:
            out = cy_expect(self.constant, matrix, self.layer_type)
            for i in range(self.n_ops):
                out += self.coefficients[i] * cy_expect(<Data> self.ops[i], matrix, self.layer_type)
        return out


cdef class CQobjFunc(CQobjEvo):
    cdef object base
    def __init__(self, base):
        self.base = base
        self.reset_shape()

    def reset_shape(self):
        self.shape = self.base.shape
        self.dims = self.base.dims
        self.issuper = self.base.issuper

    def call(self, double t, int data=0):
        return self.base(t, data=data)

    cpdef Data matmul(self, double t, Data matrix):
        cdef Data objdata = self.base(t, data=True)
        out = _data.matmul(objdata, matrix)
        return out

    cpdef Dense matmul_dense(self, double t, Dense matrix, Dense out=None):
        cdef Data objdata = self.base(t, data=True)
        if out is None:
            out = _data.matmul(objdata, matrix)
        else:
            iadd_dense(out, _data.matmul[type(objdata), Dense, Dense](objdata, matrix, 1))
        return out

    cpdef double complex expect(self, double t, Data matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        cdef double complex out
        cdef int nrow
        cdef Data objdata = self.base(t, data=True)
        if self.issuper:
            matrix = _data.column_stack(matrix)
            out = _data.expect_super(objdata, matrix)
        else:
            out = _data.expect(objdata, matrix)
        return out

    cpdef double complex expect_dense(self, double t, Dense matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        cdef double complex out
        cdef int nrow
        cdef Data objdata = self.base(t, data=True)
        if self.issuper:
            matrix = _data.column_stack(matrix)
            out = _data.expect_super(objdata, matrix)
        else:
            out = _data.expect(objdata, matrix)
        return out

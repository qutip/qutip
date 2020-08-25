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

from qutip.core.data cimport CSR, Dense, dense
from qutip.core.data.add cimport add_csr
# TODO: handle dispatch properly.  We import rather than cimport because we
# have to call with Python semantics.
from qutip.core.data.expect import (
    expect_csr, expect_super_csr, expect_csr_dense, expect_super_csr_dense,
)
from qutip.core.data.matmul cimport matmul_csr_dense_dense
from qutip.core.data.reshape cimport column_stack_csr, column_stack_dense
from qutip.core.cy.coefficient cimport Coefficient


cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)


cdef class CQobjEvo:
    """
    Dense matmul(double t, Dense matrix, Dense out=None)
      Get the matrix multiplication of self with matrix and put the result in
      `out`, if supplied.  Always returns the object it stored the output in
      (even if it was passed).

    expect(double t, CSR matrix)
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

    def call(self, double t, object coefficients=None, bint data=False):
        cdef CSR out = self.constant.copy()
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
            out = add_csr(out, self.ops[i], scale=coefficients[i])
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

    cpdef Dense matmul(self, double t, Dense matrix, Dense out=None):
        cdef size_t i
        self.dyn_args(t, matrix) # TODO: move Out
        self._factor(t)
        if out is None:
            out = matmul_csr_dense_dense(self.constant, matrix)
        else:
            matmul_csr_dense_dense(self.constant, matrix, scale=1, out=out)
        for i in range(self.n_ops):
            matmul_csr_dense_dense(self.ops[i], matrix,
                                   scale=self.coefficients[i], out=out)
        return out

    cpdef double complex expect(self, double t, Data matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        # TODO: remove shim once we have dispatching
        if self.issuper:
            matrix = (column_stack_csr(matrix) if isinstance(matrix, CSR)
                      else column_stack_dense(matrix, inplace=True))
            _expect = expect_super_csr if isinstance(matrix, CSR) else expect_super_csr_dense
        else:
            _expect = expect_csr if isinstance(matrix, CSR) else expect_csr_dense
        self.dyn_args(t, matrix) # TODO: move Out
        self._factor(t)
        # end shim
        cdef size_t i
        cdef double complex out
        out = _expect(self.constant, matrix)
        for i in range(self.n_ops):
            out += self.coefficients[i] * _expect(self.ops[i], matrix)
        return out

    def set_dyn_args(self, object dyn_args, dict args, object op):
        # Move elsewhere and op should be a dimensions object when available
        self.args = args
        self.op = op
        self.dynamic_arguments = dyn_args
        self.has_dynamic_args = bool(self.dynamic_arguments)

    cpdef dyn_args(self, double t, Data matrix):
        from ..qobjevo import dynamic_argument
        cdef Coefficient coeff
        if not self.has_dynamic_args:
            return
        else:
            for name, what, e_op in self.dynamic_arguments:
                self.args[name] = dynamic_argument(t, self.op, matrix,
                                                   what, e_op)
            for i in range(self.n_ops):
                coeff = <Coefficient> self.coeff[i]
                coeff.arguments(self.args)

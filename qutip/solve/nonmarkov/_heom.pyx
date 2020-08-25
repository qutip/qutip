#cython: language_level=3
#cython: boundscheck=False, wraparound=False
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

from libc.string cimport memcpy, memset
from qutip.core.data cimport idxint, CSR, csr

def pad_csr(CSR matrix, idxint row_scale, idxint col_scale,
            idxint insertrow=0, idxint insertcol=0):
    cdef idxint n_rows_in = matrix.shape[0]
    cdef idxint n_cols_in = matrix.shape[1]
    cdef idxint n_rows_out = n_rows_in * row_scale
    cdef idxint n_cols_out = n_cols_in * col_scale
    cdef idxint temp, ptr
    cdef size_t nnz = csr.nnz(matrix)
    cdef CSR out = csr.empty(n_rows_out, n_cols_out, nnz)

    memcpy(out.data, matrix.data, nnz * sizeof(double complex))
    if insertcol == 0:
        memcpy(out.col_index, matrix.col_index, nnz * sizeof(idxint))
    elif insertcol > 0 and insertcol < col_scale:
        temp = insertcol * n_cols_in
        for ptr in range(nnz):
            out.col_index[ptr] = matrix.col_index[ptr] + temp
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        memcpy(out.row_index, matrix.row_index, n_rows_in * sizeof(idxint))
        temp = matrix.row_index[n_rows_in]
        for ptr in range(n_rows_in, n_rows_out + 1):
            out.row_index[ptr] = temp

    elif insertrow == row_scale - 1:
        temp = insertrow * n_rows_in
        memset(out.row_index, 0, temp * sizeof(idxint))
        memcpy(out.row_index + temp, matrix.row_index,
               (n_rows_out + 1 - temp) * sizeof(idxint))

    elif insertrow > 0 and insertrow < row_scale - 1:
        temp = insertrow * n_rows_in
        memset(out.row_index, 0, temp * sizeof(idxint))
        memcpy(out.row_index + temp, matrix.row_index, n_rows_in * sizeof(idxint))
        for ptr in range(temp + n_rows_in, n_rows_out + 1):
            out.row_index[ptr] = nnz
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    return out


from qutip.core.data import Dispatcher
import inspect as _inspect

pad = Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('rowscale', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('colscale', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('insertrow', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('insertcol', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='pad',
    module=__name__,
    inputs=('matrix',),
    out=True)
pad.add_specialisations([
    (CSR, CSR, pad_csr),
])

del Dispatcher, _inspect

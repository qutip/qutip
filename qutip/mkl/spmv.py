# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation.
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
import numpy as np
import scipy.sparse as sp
import ctypes
from ctypes import POINTER,c_int,c_char,c_double, byref
from numpy import ctypeslib
import qutip.settings as qset
zcsrgemv = qset.mkl_lib.mkl_cspblas_zcsrgemv

def mkl_spmv(A, x):
     """
     sparse csr_spmv using MKL
     """
     (m,n) = A.shape

     # Pointers to data of the matrix
     data    = A.data.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))
     indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
     indices = A.indices.ctypes.data_as(POINTER(c_int))

     # Allocate output, using same conventions as input
     if x.ndim is 1:
        y = np.empty(m,dtype=np.complex,order='C')
     elif x.ndim==2 and x.shape[1]==1:
        y = np.empty((m,1),dtype=np.complex,order='C')
     else:
        raise Exception('Input vector must be 1D row or 2D column vector')
     
     np_x = x.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))
     np_y = y.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))
     # now call MKL. This returns the answer in np_y, which points to y
     zcsrgemv(byref(c_char(bytes(b'N'))), byref(c_int(m)), data ,indptr, indices, np_x, np_y ) 
     return y

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
from ctypes import POINTER, c_int, c_char, c_double, byref
mkl_lib = ctypes.cdll.LoadLibrary('/Users/paul/anaconda/lib/libmkl_rt.dylib')
from numpy import ctypeslib

# Load solver functions from mkl_lib
dss_create = mkl_lib.dss_create
dss_define_structure = mkl_lib.dss_define_structure
dss_reorder = mkl_lib.dss_reorder
dss_factor_complex = mkl_lib.dss_factor_complex
dss_solve_complex = mkl_lib.dss_solve_complex

# Set solver constant parameters
MKL_DSS_ZERO_BASED_INDEXING = 131072
MKL_DSS_AUTO_REORDER = 268435520
MKL_DSS_MY_ORDER = 268435584
MKL_DSS_OPTION1_ORDER = 536871104

MKL_DSS_MSG_LVL_WARNING = -2147483644
MKL_DSS_MSG_TERM_LVL_ERROR = 1073741864

MKL_DSS_SYMMETRIC = 536870976
MKL_DSS_SYMMETRIC_STRUCTURE = 536871040
MKL_DSS_NON_SYMMETRIC = 536871104
MKL_DSS_SYMMETRIC_COMPLEX = 536871168
MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX = 536871232
MKL_DSS_NON_SYMMETRIC_COMPLEX = 536871296

MKL_DSS_POSITVE_DEFINITE = 134217792
MKL_DSS_INDEFINITE = 134217856
MKL_DSS_HERMITIAN_POSITIVE_DEFINITE = 134217920
MKL_DSS_HERMITIAN_INDEFINITE = 134217984


def mkl_spsolve(A, b):
    dim = A.shape[0]
    #Create working array and pointer to array
    pt = np.zeros(64, dtype=int, order='C')
    np_pt = pt.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    
    # Options for solver initialization
    create_opts = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_MSG_TERM_LVL_ERROR + \
                MKL_DSS_NON_SYMMETRIC_COMPLEX + MKL_DSS_AUTO_REORDER + \
                MKL_DSS_ZERO_BASED_INDEXING
    
    # Init solver
    create_status = dss_create(np_pt, byref(c_int(create_opts)))
    if create_status != 0:
        print(create_status)
    
    # Create pointers to sparse matrix arrays
    data = A.data.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C')) 
    indptr = A.indptr.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    indices = A.indices.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    nnz = A.nnz
    
    # Create solution array (x) and pointers to x and b
    x = np.zeros(dim, dtype=complex, order='C')
    np_x = x.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C')) 
    np_b = b.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))
    
    # Create perm array and pointer
    perm = np.zeros(dim, dtype=int, order='C')
    np_perm = perm.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
       
    # Options for define structure
    structure_opts = MKL_DSS_NON_SYMMETRIC_COMPLEX
    
    # Define sparse structure
    structure_status = dss_define_structure(np_pt, byref(c_int(structure_opts)),
                        indptr, byref(c_int(dim)), byref(c_int(dim)), indices,
                        byref(c_int(nnz)))
    if structure_status != 0:
        print(structure_status)
    
    # Define reorder opts
    reorder_opts = MKL_DSS_AUTO_REORDER
    
    # Reorder sparse matrix
    reorder_status = dss_reorder(np_pt, byref(c_int(reorder_opts)), np_perm)
    if reorder_status != 0:
        print(reorder_status)
    
    #Define factoring opts
    factor_opts = MKL_DSS_INDEFINITE
    
    # Factor matrix
    factor_status = dss_factor_complex(np_pt, byref(c_int(factor_opts)), data)
    if factor_status != 0:
        print(factor_status)

    # Define soolving phase opts
    nrhs = 1
    solve_opts = 0
    
    # Perform solve
    solve_status = dss_solve_complex(np_pt, byref(c_int(solve_opts)), np_b,
                    byref(c_int(nrhs)), np_x)
    if solve_status != 0:
        print(solve_status)
    
    #Return solution vector
    return x


if __name__ == "__main__":
    from qutip import *
    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]

    rho_ss = steadystate(H, c_ops)
    rho_ss_analytic = coherent_dm(N, -1.0j * (Omega)/(Gamma/2))

    L = liouvillian(H, c_ops)
    dims = L.dims[0]
    n = int(np.sqrt(L.shape[0]))
    L = L.data + sp.csr_matrix(
        (np.ones(n), (np.zeros(n), [nn * (n + 1)
         for nn in range(n)])),
        shape=(n ** 2, n ** 2))
    b = np.zeros(n ** 2, dtype=complex)
    b[0] = 1
    x = mkl_spsolve(L, b)

    data = vec2mat(x)
    data = 0.5 * (data + data.conj().T)
    rho_ss2 = Qobj(data, dims=dims, isherm=True)

    print(rho_ss-rho_ss2)
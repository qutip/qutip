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
from ctypes import POINTER, c_int, c_char, c_char_p, c_double, byref
from numpy import ctypeslib
import qutip.settings as qset

# Load solver functions from mkl_lib
dss_create = qset.mkl_lib.dss_create
dss_define_structure = qset.mkl_lib.dss_define_structure
dss_reorder = qset.mkl_lib.dss_reorder
dss_factor_real = qset.mkl_lib.dss_factor_real
dss_factor_complex = qset.mkl_lib.dss_factor_complex
dss_solve_real = qset.mkl_lib.dss_solve_real
dss_solve_complex = qset.mkl_lib.dss_solve_complex
dss_statistics = qset.mkl_lib.dss_statistics
dss_delete = qset.mkl_lib.dss_delete

# Set solver constant parameters
# Create parameters
MKL_DSS_DEFAULTS                    = 0
MKL_DSS_ZERO_BASED_INDEXING         = 131072

MKL_DSS_MSG_LVL_SUCCESS             = -2147483647
MKL_DSS_MSG_LVL_DEBUG               = -2147483646
MKL_DSS_MSG_LVL_INFO                = -2147483645
MKL_DSS_MSG_LVL_WARNING             = -2147483644
MKL_DSS_MSG_LVL_ERROR               = -2147483643

MKL_DSS_TERM_LVL_SUCCESS            = 1073741832
MKL_DSS_TERM_LVL_DEBUG              = 1073741840
MKL_DSS_TERM_LVL_INFO               = 1073741848
MKL_DSS_TERM_LVL_WARNING            = 1073741856
MKL_DSS_TERM_LVL_ERROR              = 1073741864
MKL_DSS_TERM_LVL_FATAL              = 1073741872

# Structure parameters
MKL_DSS_SYMMETRIC                   = 536870976
MKL_DSS_SYMMETRIC_STRUCTURE         = 536871040
MKL_DSS_NON_SYMMETRIC               = 536871104
MKL_DSS_SYMMETRIC_COMPLEX           = 536871168
MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX = 536871232
MKL_DSS_NON_SYMMETRIC_COMPLEX       = 536871296

# Reorder parameters
MKL_DSS_AUTO_REORDER                = 268435520
MKL_DSS_METIS_OPENMP_ORDER          = 268435840
MKL_DSS_MY_ORDER                    = 268435584
MKL_DSS_GET_ORDER                   = 268435712
MKL_DSS_OPTION1_ORDER               = 536871104

# Factor parameters
MKL_DSS_POSITVE_DEFINITE            = 134217792
MKL_DSS_INDEFINITE                  = 134217856
MKL_DSS_HERMITIAN_POSITIVE_DEFINITE = 134217920
MKL_DSS_HERMITIAN_INDEFINITE        = 134217984

# Set error messages
dss_error_msgs = {'0' : 'Success', '-1': 'Zero Pivot', '-2': 'Out of memory', '-3': 'Failure',
                '-4' : 'Row error', '-5': 'Column error', '-6': 'Too few values',
                '-7': 'Too many values', '-8': 'Not square matrix', '-9': 'State error',
                '-10': 'Invalid option', '-11': 'Option conflict', '-12': 'MSG level error',
                '-13': 'TERM level error', '-14': 'Structure error', '-15': 'Reorder error',
                '-16': 'Values error', '-17': 'Statistics invalid matrix', 
                '-18': 'Statistics invalid state', '-19': 'Statistics invalid string'}


def _default_solver_args():
    def_args = {'hermitian': False, 'symmetric': False, 'posdef': False}
    return def_args



def mkl_spsolve(A, b, perm=None, verbose=False, **kwargs):
    
    if not sp.isspmatrix_csr(A):
        raise TypeError('Input matrix must be in sparse CSR format.')
    
    ndim = b.ndim
    if ndim == 2 and b.shape[0] == 1:
        b = b.ravel()
        ndim = 1
    
    if A.shape[0] != A.shape[1]:
        raise Exception('Input matrix must be square')
    
    solver_args = _default_solver_args()
    for key in kwargs.keys():
        if key in solver_args.keys():
            solver_args[key] = kwargs[key]
        else:
            raise Exception(
                "Invalid keyword argument '"+key+"' passed to mkl_spsolve.")
    
    dim = A.shape[0]
    if A.dtype == np.complex128:
        is_complex = 1
        data_type = np.complex128
        if b.dtype != np.complex128:
            b = b.astype(np.complex128, copy=False)
    else:
        is_complex = 0
        data_type = np.float64
        if A.dtype != np.float64:
            A = sp.csr_matrix(A, dtype=float, copy=False)
        if b.dtype != np.float64:
            b = b.astype(np.float64, copy=False)
    
    #Create working array and pointer to array
    pt = np.zeros(64, dtype=int, order='C')
    np_pt = pt.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    
    # Options for solver initialization
    create_opts = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + \
                MKL_DSS_ZERO_BASED_INDEXING
    
    # Init solver
    create_status = dss_create(np_pt, byref(c_int(create_opts)))
    if create_status != 0:
        raise Exception(dss_error_msgs[str(create_status)])
    
    # Create pointers to sparse matrix arrays
    data = A.data.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=1, flags='C')) 
    indptr = A.indptr.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    indices = A.indices.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    nnz = A.nnz
    
    # Create solution array (x) and pointers to x and b
    if is_complex:
        x = np.zeros(b.shape, dtype=np.complex128, order='C')
    else:
        x = np.zeros(b.shape, dtype=np.float64, order='C')
    np_x = x.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=ndim, flags='C')) 
    np_b = b.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=ndim, flags='C'))
       
    #Define array for statistics
    stat_opts = MKL_DSS_DEFAULTS
    stats = np.zeros(6, dtype=np.float64, order='C')
    np_stats = stats.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=1, flags='C'))
    
    # Options for define structure
    if is_complex:
        if solver_args['hermitian']:
            structure_opts = MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX
        elif solver_args['symmetric']:
            structure_opts = MKL_DSS_SYMMETRIC_COMPLEX
        else:
            structure_opts = MKL_DSS_NON_SYMMETRIC_COMPLEX
    else:
        if solver_args['symmetric']:
            structure_opts = MKL_DSS_SYMMETRIC
        else:
            structure_opts = MKL_DSS_NON_SYMMETRIC
    
    # Define sparse structure
    status = dss_define_structure(np_pt, byref(c_int(structure_opts)),
                        indptr, byref(c_int(dim)), byref(c_int(dim)), indices,
                        byref(c_int(nnz)))
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    
    
    # Create perm array and pointer
    if perm is None:
        perm = np.zeros(dim, dtype=np.int32, order='C')
        reorder_opts = MKL_DSS_METIS_OPENMP_ORDER
    else:
        reorder_opts = MKL_DSS_MY_ORDER
    np_perm = perm.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    
    # Reorder sparse matrix
    status = dss_reorder(np_pt, byref(c_int(reorder_opts)), np_perm)
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    
    # Get reordering statistics
    stat_string = c_char_p(bytes(b'ReorderTime,Peakmem,Factormem'))
    status = dss_statistics(np_pt, byref(c_int(stat_opts)), stat_string, np_stats)
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    if verbose:
        print('Reordering Phase')
        print('----------------')
        print('Reorder time:',stats[0])
        print('Peak memory (Mb):',stats[1]/1024.)
        print('Factor memory (Mb):',stats[2]/1024.)
        print('')
    
    #Define factoring opts
    if is_complex:
        if solver_args['hermitian']:
            if solver_args['posdef']:
                factor_opts = MKL_DSS_HERMITIAN_POSITIVE_DEFINITE
            else:
                factor_opts = MKL_DSS_HERMITIAN_INDEFINITE
        else:
            factor_opts = MKL_DSS_INDEFINITE
    else:
        if solver_args['posdef']:
            factor_opts = MKL_DSS_POSITVE_DEFINITE
        else:
            factor_opts = MKL_DSS_INDEFINITE
    
    # Factor matrix
    if is_complex:
        status = dss_factor_complex(np_pt, byref(c_int(factor_opts)), data)
    else:
        status = dss_factor_real(np_pt, byref(c_int(factor_opts)), data)
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])

    # Get factorization statistics
    stat_string = c_char_p(bytes(b'FactorTime,Flops'))
    status = dss_statistics(np_pt, byref(c_int(stat_opts)), stat_string, np_stats)
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    if verbose:
        print('Factorization Phase')
        print('-------------------')
        print('Factor time:',stats[0])
        print('Flops:',stats[1])
        print('')
    
    
    # Define solving phase opts
    if ndim == 2:
        nrhs = b.shape[0]
    else:
        nrhs = 1
    
    # Dont do any iterative refinement.
    solve_opts = MKL_DSS_DEFAULTS
    
    # Perform solve
    if is_complex:
        status = dss_solve_complex(np_pt, byref(c_int(solve_opts)), np_b,
                    byref(c_int(nrhs)), np_x)
    else:
        status = dss_solve_real(np_pt, byref(c_int(solve_opts)), np_b,
                    byref(c_int(nrhs)), np_x)
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    
    # Get all statistics
    stat_string = c_char_p(bytes(b'ReorderTime,FactorTime,SolveTime,Peakmem,Factormem,Solvemem'))
    status = dss_statistics(np_pt, byref(c_int(stat_opts)), stat_string, np_stats)
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    if verbose:
        print('Solution Phase')
        print('--------------')
        print('Solve time:',stats[2])
        print('Solve memory (Mb):',stats[5]/1024.)
    
    
    # Delete opts
    delete_opts = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR
    status = dss_delete(np_pt, byref(c_int(delete_opts)))
    if status != 0:
        raise Exception(dss_error_msgs[str(status)])
    
    #Return solution vector
    return x


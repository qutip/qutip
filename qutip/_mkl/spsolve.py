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
from __future__ import print_function, division
import sys
import numpy as np
import scipy.sparse as sp
import ctypes
from ctypes import POINTER, c_int, c_char, c_char_p, c_double, byref
from numpy import ctypeslib
import time
import qutip.settings as qset

# Load solver functions from mkl_lib
pardiso = qset.mkl_lib.pardiso
pardiso_delete = qset.mkl_lib.pardiso_handle_delete
if sys.maxsize > 2**32: #Running 64-bit
    pardiso_64 = qset.mkl_lib.pardiso_64
    pardiso_delete_64 = qset.mkl_lib.pardiso_handle_delete_64


def _pardiso_parameters(hermitian=False, has_perm=False):
    iparm = np.zeros(64, dtype=np.int32)
    iparm[0] = 1 # Do not use default values
    iparm[1] = 3 # Use openmp nested dissection
    if has_perm:
        iparm[4] = 1
    iparm[7] = 20 # Max number of iterative refinements
    if hermitian:
        iparm[9] = 8
    else:
        iparm[9] = 13 
    if not hermitian:
        iparm[10] = 1 # Scaling vectors
        iparm[12] = 0 # Do not use non-symmetric weighted matching
    iparm[17] = -1
    iparm[20] = 1
    iparm[23] = 1 # Parallel factorization
    iparm[26] = 0 # Check matrix structure
    iparm[34] = 1 # Use zero-based indexing
    return iparm
    
    

# Set error messages
pardiso_error_msgs = {'-1': 'Input inconsistant', '-2': 'Out of memory', '-3': 'Reordering problem',
                '-4' : 'Zero pivot, numerical factorization or iterative refinement problem', 
                '-5': 'Unclassified internal error', '-6': 'Reordering failed',
                '-7': 'Diagonal matrix is singular', '-8': '32-bit integer overflow', 
                '-9': 'Not enough memeory for OOC', '-10': 'Error opening OOC files', 
                '-11': 'Read/write error with OOC files', 
                '-12': 'Pardiso-64 called from 32-bit library'}
                
def _default_solver_args():
    def_args = {'hermitian': False, 'posdef': False, 'return_info': False}
    return def_args


class mkl_lu(object):
    """
    Object pointing to LU factorization of a sparse matrix
    generated by mkl_splu.
    
    Methods
    -------
    solve(b, verbose=False)
        Solve system of equations using given RHS vector 'b'.
        Returns solution ndarray with same shape as input.
    
    info()
        Returns the statistics of the factorization and 
        solution in the lu.info attribute.
    
    delete()
        Deletes the allocated solver memory.
    
    """
    def __init__(self, np_pt=None, dim=None, is_complex=None, 
                data=None, indptr=None, indices=None,
                iparm=None, np_iparm=None, mtype=None, perm=None,
                np_perm=None, factor_time=None):
        self._np_pt = np_pt
        self._dim = dim
        self._is_complex = is_complex
        self._data = data
        self._indptr = indptr
        self._indices = indices
        self._iparm = iparm
        self._np_iparm = np_iparm
        self._mtype = mtype
        self._perm = perm
        self._np_perm = np_perm
        self._factor_time = factor_time
        self._solve_time = None
    
    def solve(self, b, verbose = None):
        b_shp = b.shape
        if b.ndim == 2 and b.shape[1] == 1:
            b = b.ravel()
            nrhs = 1
        
        elif b.ndim == 2 and b.shape[1] != 1:
            nrhs = b.shape[1]
            b = b.ravel(order='F')
        
        else:
            b = b.ravel()
            nrhs = 1

        if self._is_complex:
            data_type = np.complex128
            if b.dtype != np.complex128:
                b = b.astype(np.complex128, copy=False)
        else:
            data_type = np.float64
            if b.dtype != np.float64:
                b = b.astype(np.float64, copy=False)

        # Create solution array (x) and pointers to x and b
        if self._is_complex:
            x = np.zeros(b.shape, dtype=np.complex128, order='C')
        else:
            x = np.zeros(b.shape, dtype=np.float64, order='C')
        
        np_x = x.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=1, flags='C')) 
        np_b = b.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=1, flags='C'))
        
        error = np.zeros(1,dtype=np.int32)
        np_error = error.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C')) 
        #Call solver
        _solve_start = time.time()
        pardiso(self._np_pt, byref(c_int(1)), byref(c_int(1)), byref(c_int(self._mtype)),
            byref(c_int(33)), byref(c_int(self._dim)), self._data, self._indptr, self._indices, 
            self._np_perm, byref(c_int(nrhs)), self._np_iparm, byref(c_int(0)), np_b,
            np_x, np_error)
        self._solve_time = time.time() -_solve_start
        if error[0] != 0:
            raise Exception(pardiso_error_msgs[str(error[0])])
        
        if verbose:
            print('Solution Stage')
            print('--------------')
            print('Solution time:                  ',round(self._solve_time,4))
            print('Solution memory (Mb):           ',round(self._iparm[16]/1024.,4))
            print('Number of iterative refinements:',self._iparm[6])
            print('Total memory (Mb):              ',round(sum(self._iparm[15:17])/1024.,4))
            print()
        
        # Return solution vector x
        if nrhs==1:
            if x.shape != b_shp:
                x = np.reshape(x, b_shp)
            return x
        else:
            return np.reshape(x, b_shp, order='F')
    
    def info(self):
        info = {'FactorTime': self._factor_time,
                'SolveTime': self._solve_time,
                'Factormem': round(self._iparm[15]/1024.,4), 
                'Solvemem': round(self._iparm[16]/1024.,4),
                'IterRefine': self._iparm[6]}
        return info
        
    
    def delete(self):
        #Delete all data
        error = np.zeros(1,dtype=np.int32)
        np_error = error.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
        pardiso(self._np_pt, byref(c_int(1)), byref(c_int(1)), byref(c_int(self._mtype)),
            byref(c_int(-1)), byref(c_int(self._dim)), self._data, self._indptr, self._indices, 
            self._np_perm, byref(c_int(1)), self._np_iparm, byref(c_int(0)), byref(c_int(0)),
            byref(c_int(0)), np_error)
        if error[0] == -10:
            raise Exception('Error freeing solver memory')
        
        
        

def mkl_splu(A, perm=None, verbose=False, **kwargs):
    """
    Returns the LU factorization of the sparse matrix A.
    
    Parameters
    ----------
    A : csr_matrix
        Sparse input matrix.
    perm : ndarray (optional)
        User defined matrix factorization permutation.
    verbose : bool {False, True}
        Report factorization details.
    
    Returns
    -------
    lu : mkl_lu
        Returns object containing LU factorization with a
        solve method for solving with a given RHS vector.
    
    """
    if not sp.isspmatrix_csr(A):
        raise TypeError('Input matrix must be in sparse CSR format.')
    
    if A.shape[0] != A.shape[1]:
        raise Exception('Input matrix must be square')
    
    dim = A.shape[0]
    solver_args = _default_solver_args()
    for key in kwargs.keys():
        if key in solver_args.keys():
            solver_args[key] = kwargs[key]
        else:
            raise Exception(
                "Invalid keyword argument '"+key+"' passed to mkl_splu.")
    
    # If hermitian, then take upper-triangle of matrix only
    if solver_args['hermitian']:
        B = sp.triu(A, format='csr')
        A = B #This gets around making a full copy of A in triu
    if A.dtype == np.complex128:
        is_complex = 1
        data_type = np.complex128
    else:
        is_complex = 0
        data_type = np.float64
        if A.dtype != np.float64:
            A = sp.csr_matrix(A, dtype=np.float64, copy=False)
        
    #Create pointer to internal memory
    pt = np.zeros(64,dtype=int)
    np_pt = pt.ctypes.data_as(ctypeslib.ndpointer(int, ndim=1, flags='C'))
    
    # Create pointers to sparse matrix arrays
    data = A.data.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=1, flags='C')) 
    indptr = A.indptr.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    indices = A.indices.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    nnz = A.nnz
    
    # Setup perm array
    if perm is None:
        perm = np.zeros(dim, dtype=np.int32)
        has_perm = 0
    else:
        has_perm = 1
    np_perm = perm.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    
    # setup iparm 
    iparm = _pardiso_parameters(solver_args['hermitian'], has_perm)
    np_iparm = iparm.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    
    # setup call parameters
    matrix_dtype = ''
    matrix_type = ''
    
    if data_type == np.complex128:
        matrix_dtype = 'Complex '
        if solver_args['hermitian']:
            if solver_args['posdef']:
                mtype = 4
                matrix_type = 'Hermitian postive-definite'
            else:
                mtype = -4
                matrix_type = 'Hermitian indefinite'
        else:
            mtype = 13
            matrix_type = 'Non-symmetric'
    else:
        matrix_dtype = 'Real '
        if solver_args['hermitian']:
            if solver_args['posdef']:
                mtype = 2 
                matrix_type = 'symmetric postive-definite'
            else:
                 mtype = -2 
                 matrix_type = 'symmetric indefinite'
        else:
            mtype = 11
            matrix_type = 'Non-symmetric'
    
    if verbose:
        print('Solver Initialization')
        print('---------------------')
        print('Input matrix type: ', matrix_dtype+matrix_type)
        print('Input matrix shape:', A.shape)
        print('Input matrix NNZ:  ', A.nnz)
        print()
        
    b =  np.zeros(1, dtype=data_type) # Input dummy RHS at this phase
    np_b = b.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=1, flags='C'))
    x =  np.zeros(1, dtype=data_type) # Input dummy solution at this phase
    np_x = x.ctypes.data_as(ctypeslib.ndpointer(data_type, ndim=1, flags='C'))
    
    error = np.zeros(1,dtype=np.int32)
    np_error = error.ctypes.data_as(ctypeslib.ndpointer(np.int32, ndim=1, flags='C'))
    
    #Call solver
    _factor_start = time.time()
    pardiso(np_pt, byref(c_int(1)), byref(c_int(1)), byref(c_int(mtype)),
            byref(c_int(12)), byref(c_int(dim)), data, indptr, indices, np_perm,
            byref(c_int(1)), np_iparm, byref(c_int(0)), np_b,
            np_x, np_error)
    _factor_time = time.time() - _factor_start
    if error[0] != 0:
        raise Exception(pardiso_error_msgs[str(error[0])])
    
    if verbose:
        print('Analysis and Factorization Stage')
        print('--------------------------------')
        print('Factorization time:       ',round(_factor_time,4))
        print('Factorization memory (Mb):',round(iparm[15]/1024.,4))
        print('NNZ in LU factors:        ',iparm[17])
        print()
    
    return mkl_lu(np_pt, dim, is_complex, data, indptr, indices, 
                  iparm, np_iparm, mtype, perm, np_perm, _factor_time)


def mkl_spsolve(A, b, perm=None, verbose=False, **kwargs):
    """
    Solves a sparse linear system of equations using the 
    Intel MKL Pardiso solver.
    
    Parameters
    ----------
    A : csr_matrix
        Sparse matrix.
    b : ndarray or sparse matrix
        The vector or matrix representing the right hand side of the equation.
        If a vector, b.shape must be (n,) or (n, 1).
    perm : ndarray (optional)
        User defined matrix factorization permutation.
    
    Returns
    -------
    x : ndarray or csr_matrix
        The solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])
    
    """
    lu = mkl_splu(A, perm=perm, verbose=verbose, **kwargs)
    b_is_sparse = sp.isspmatrix(b)
    b_shp = b.shape
    if b_is_sparse and b.shape[1] == 1:
        b = b.toarray()
        b_is_sparse = False
    elif b_is_sparse and b.shape[1] != 1:
        nrhs = b.shape[1]
        if lu._is_complex:
            b = sp.csc_matrix(b, dtype=np.complex128, copy=False)
        else:
            b = sp.csc_matrix(b, dtype=np.float64, copy=False)
    
    # Do dense RHS solving
    if not b_is_sparse:
        x = lu.solve(b, verbose=verbose)
    # Solve each RHS vec individually and convert to sparse 
    else:
        data_segs = []
        row_segs = []
        col_segs = []
        for j in range(nrhs):
            bj = b[:, j].A.ravel()
            xj = lu.solve(bj)
            w = np.flatnonzero(xj)
            segment_length = w.shape[0]
            row_segs.append(w)
            col_segs.append(np.ones(segment_length, dtype=np.int32)*j)
            data_segs.append(np.asarray(xj[w], dtype=xj.dtype))
        sp_data = np.concatenate(data_segs)
        sp_row = np.concatenate(row_segs)
        sp_col = np.concatenate(col_segs)
        x = sp.coo_matrix((sp_data,(sp_row,sp_col)),
                        shape=b_shp).tocsr()
    
    info = lu.info()
    lu.delete()
    if 'return_info' in kwargs.keys() and kwargs['return_info'] == True:
        return x, info
    else:    
        return x

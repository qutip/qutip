# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
import operator
from scipy.sparse import (_sparsetools, isspmatrix, isspmatrix_csr,
                          csr_matrix, coo_matrix, csc_matrix, dia_matrix)
from scipy.sparse.sputils import (upcast, upcast_char, to_native, isdense, isshape,
                      getdtype, isscalarlike, IndexMixin, get_index_dtype,
                      downcast_intp_index, get_sum_dtype)
from scipy.sparse.base import spmatrix, isspmatrix, SparseEfficiencyWarning
from warnings import warn

class fast_csr_matrix(csr_matrix):
    """
    A subclass of scipy.sparse.csr_matrix that skips the data format
    checks that are run everytime a new csr_matrix is created.
    """
    def __init__(self, args=None, shape=None, dtype=None, copy=False):
        if args is None: #Build zero matrix
            if shape is None:
                raise Exception('Shape must be given when building zero matrix.')
            self.data = np.array([], dtype=complex)
            self.indices = np.array([], dtype=np.int32)
            self.indptr = np.zeros(shape[0]+1, dtype=np.int32)
            self._shape = tuple(shape)
            
        else:
            if args[0].shape[0] and args[0].dtype != complex:
                raise TypeError('fast_csr_matrix allows only complex data.')
            if args[1].shape[0] and args[1].dtype != np.int32:
                raise TypeError('fast_csr_matrix allows only int32 indices.')
            if args[2].shape[0] and args[1].dtype != np.int32:
                raise TypeError('fast_csr_matrix allows only int32 indptr.')
            self.data = np.array(args[0], dtype=complex, copy=copy)
            self.indices = np.array(args[1], dtype=np.int32, copy=copy)
            self.indptr = np.array(args[2], dtype=np.int32, copy=copy)
            if shape is None:
                self._shape = tuple([len(self.indptr)-1]*2)
            else:
                self._shape = tuple(shape)
        self.dtype = complex
        self.maxprint = 50
        self.format = 'csr'

    def _binopt(self, other, op):
        """
        Do the binary operation fn to two sparse matrices using 
        fast_csr_matrix only when other is also a fast_csr_matrix.
        """
        # e.g. csr_plus_csr, csr_minus_csr, etc.
        if not isinstance(other, fast_csr_matrix):
            other = csr_matrix(other)
        # e.g. csr_plus_csr, csr_minus_csr, etc.
        fn = getattr(_sparsetools, self.format + op + self.format)

        maxnnz = self.nnz + other.nnz
        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=maxnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(maxnnz, dtype=idx_dtype)

        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(maxnnz, dtype=np.bool_)
        else:
            data = np.empty(maxnnz, dtype=upcast(self.dtype, other.dtype))

        fn(self.shape[0], self.shape[1],
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        actual_nnz = indptr[-1]
        indices = indices[:actual_nnz]
        data = data[:actual_nnz]
        if actual_nnz < maxnnz // 2:
            # too much waste, trim arrays
            indices = indices.copy()
            data = data.copy()
        if isinstance(other, fast_csr_matrix) and (not op in bool_ops):
            A = fast_csr_matrix((data, indices, indptr), dtype=data.dtype, shape=self.shape)
        else:
            A = csr_matrix((data, indices, indptr), dtype=data.dtype, shape=self.shape)
        return A
    
    def multiply(self, other):
        """Point-wise multiplication by another matrix, vector, or
        scalar.
        """
        # Scalar multiplication.
        if isscalarlike(other):
            return self._mul_scalar(other)
        # Sparse matrix or vector.
        if isspmatrix(other):
            if self.shape == other.shape:
                if not isinstance(other, fast_csr_matrix):
                    other = csr_matrix(other)
                return self._binopt(other, '_elmul_')
            # Single element.
            elif other.shape == (1,1):
                return self._mul_scalar(other.toarray()[0, 0])
            elif self.shape == (1,1):
                return other._mul_scalar(self.toarray()[0, 0])
            # A row times a column.
            elif self.shape[1] == other.shape[0] and self.shape[1] == 1:
                return self._mul_sparse_matrix(other.tocsc())
            elif self.shape[0] == other.shape[1] and self.shape[0] == 1:
                return other._mul_sparse_matrix(self.tocsc())
            # Row vector times matrix. other is a row.
            elif other.shape[0] == 1 and self.shape[1] == other.shape[1]:
                other = dia_matrix((other.toarray().ravel(), [0]),
                                    shape=(other.shape[1], other.shape[1]))
                return self._mul_sparse_matrix(other)
            # self is a row.
            elif self.shape[0] == 1 and self.shape[1] == other.shape[1]:
                copy = dia_matrix((self.toarray().ravel(), [0]),
                                    shape=(self.shape[1], self.shape[1]))
                return other._mul_sparse_matrix(copy)
            # Column vector times matrix. other is a column.
            elif other.shape[1] == 1 and self.shape[0] == other.shape[0]:
                other = dia_matrix((other.toarray().ravel(), [0]),
                                    shape=(other.shape[0], other.shape[0]))
                return other._mul_sparse_matrix(self)
            # self is a column.
            elif self.shape[1] == 1 and self.shape[0] == other.shape[0]:
                copy = dia_matrix((self.toarray().ravel(), [0]),
                                    shape=(self.shape[0], self.shape[0]))
                return copy._mul_sparse_matrix(other)
            else:
                raise ValueError("inconsistent shapes")
        # Dense matrix.
        if isdense(other):
            if self.shape == other.shape:
                ret = self.tocoo()
                ret.data = np.multiply(ret.data, other[ret.row, ret.col]
                                       ).view(np.ndarray).ravel()
                return ret
            # Single element.
            elif other.size == 1:
                return self._mul_scalar(other.flat[0])
        # Anything else.
        return np.multiply(self.todense(), other)
    
    def _mul_sparse_matrix(self, other):
        """
        Do the sparse matrix mult returning fast_csr_matrix only
        when other is also fast_csr_matrix.
        """
        M, K1 = self.shape
        K2, N = other.shape

        major_axis = self._swap((M,N))[0]
        if not isinstance(other, fast_csr_matrix):
            other = csr_matrix(other)  # convert to this format
        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=M*N)
        indptr = np.empty(major_axis + 1, dtype=idx_dtype)

        fn = getattr(_sparsetools, self.format + '_matmat_pass1')
        fn(M, N,
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           indptr)

        nnz = indptr[-1]
        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=nnz)
        indptr = np.asarray(indptr, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

        fn = getattr(_sparsetools, self.format + '_matmat_pass2')
        fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)
        if isinstance(other, fast_csr_matrix):
            return fast_csr_matrix((data,indices,indptr), shape=(M,N))
        else:
            return csr_matrix((data,indices,indptr),shape=(M,N))

    def _scalar_binopt(self, other, op):
        """Scalar version of self._binopt, for cases in which no new nonzeros
        are added. Produces a new spmatrix in canonical form.
        """
        self.sum_duplicates()
        res = self._with_data(op(self.data, other), copy=True)
        res.eliminate_zeros()
        return res

    def __eq__(self, other):
        # Scalar other.
        if isscalarlike(other):
            if np.isnan(other):
                return csr_matrix(self.shape, dtype=np.bool_)

            if other == 0:
                warn("Comparing a sparse matrix with 0 using == is inefficient"
                        ", try using != instead.", SparseEfficiencyWarning)
                all_true = _all_true(self.shape)
                inv = self._scalar_binopt(other, operator.ne)
                return all_true - inv
            else:
                return self._scalar_binopt(other, operator.eq)
        # Dense other.
        elif isdense(other):
            return self.todense() == other
        # Sparse other.
        elif isspmatrix(other):
            warn("Comparing sparse matrices using == is inefficient, try using"
                    " != instead.", SparseEfficiencyWarning)
            #TODO sparse broadcasting
            if self.shape != other.shape:
                return False
            elif self.format != other.format:
                other = other.asformat(self.format)
            res = self._binopt(other,'_ne_')
            all_true = _all_true(self.shape)
            return all_true - res
        else:
            return False

    def __ne__(self, other):
        # Scalar other.
        if isscalarlike(other):
            if np.isnan(other):
                warn("Comparing a sparse matrix with nan using != is inefficient",
                     SparseEfficiencyWarning)
                all_true = _all_true(self.shape)
                return all_true
            elif other != 0:
                warn("Comparing a sparse matrix with a nonzero scalar using !="
                     " is inefficient, try using == instead.", SparseEfficiencyWarning)
                all_true = _all_true(self.shape)
                inv = self._scalar_binopt(other, operator.eq)
                return all_true - inv
            else:
                return self._scalar_binopt(other, operator.ne)
        # Dense other.
        elif isdense(other):
            return self.todense() != other
        # Sparse other.
        elif isspmatrix(other):
            #TODO sparse broadcasting
            if self.shape != other.shape:
                return True
            elif self.format != other.format:
                other = other.asformat(self.format)
            return self._binopt(other,'_ne_')
        else:
            return True

    def _inequality(self, other, op, op_name, bad_scalar_msg):
        # Scalar other.
        if isscalarlike(other):
            if 0 == other and op_name in ('_le_', '_ge_'):
                raise NotImplementedError(" >= and <= don't work with 0.")
            elif op(0, other):
                warn(bad_scalar_msg, SparseEfficiencyWarning)
                other_arr = np.empty(self.shape, dtype=np.result_type(other))
                other_arr.fill(other)
                other_arr = csr_matrix(other_arr)
                return self._binopt(other_arr, op_name)
            else:
                return self._scalar_binopt(other, op)
        # Dense other.
        elif isdense(other):
            return op(self.todense(), other)
        # Sparse other.
        elif isspmatrix(other):
            #TODO sparse broadcasting
            if self.shape != other.shape:
                raise ValueError("inconsistent shapes")
            elif self.format != other.format:
                other = other.asformat(self.format)
            if op_name not in ('_ge_', '_le_'):
                return self._binopt(other, op_name)

            warn("Comparing sparse matrices using >= and <= is inefficient, "
                 "using <, >, or !=, instead.", SparseEfficiencyWarning)
            all_true = _all_true(self.shape)
            res = self._binopt(other, '_gt_' if op_name == '_le_' else '_lt_')
            return all_true - res
        else:
            raise ValueError("Operands could not be compared.")
        
    def _with_data(self,data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        # We need this just in case something like abs(data) gets called
        # does nothing if data.dtype is complex.
        data = np.asarray(data, dtype=complex)
        if copy:
            return fast_csr_matrix((data,self.indices.copy(),self.indptr.copy()),
                                   shape=self.shape,dtype=data.dtype)
        else:
            return fast_csr_matrix((data,self.indices,self.indptr),
                                   shape=self.shape,dtype=data.dtype)
                                   
    def trans(self):
        """
        Returns the transpose of the matrix, keeping
        it in fast_csr format.
        """
        return zcsr_transpose(self)
    
    def adjoint(self):
        """
        Returns the conjugate-transpose of the matrix, keeping
        it in fast_csr format.
        """
        return zcsr_adjoint(self)
    

#Convenience funtions
#--------------------
def _all_true(shape):
    A = csr_matrix((np.ones(np.prod(shape), dtype=np.bool_),
                np.tile(np.arange(shape[1],dtype=np.int32),shape[0]),
                np.arange(0,np.prod(shape)+1,shape[1],dtype=np.int32)),
                shape=shape)
    return A


#Need to do some trailing imports here
#-------------------------------------
from qutip.cy.spmath import (zcsr_transpose, zcsr_adjoint)
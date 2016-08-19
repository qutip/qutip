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
"""
This module contains a collection of graph theory routines used mainly
to reorder matrices for iterative steady state solvers.
"""

__all__ = ['graph_degree', 'column_permutation', 'breadth_first_search', 
            'reverse_cuthill_mckee', 'maximum_bipartite_matching', 
            'weighted_bipartite_matching']

import numpy as np
import scipy.sparse as sp
from qutip.cy.graph_utils import (
    _breadth_first_search, _node_degrees,
    _reverse_cuthill_mckee, _maximum_bipartite_matching,
    _weighted_bipartite_matching)


def graph_degree(A):
    """
    Returns the degree for the nodes (rows) of a symmetric
    graph in sparse CSR or CSC format, or a qobj.

    Parameters
    ----------
    A : qobj, csr_matrix, csc_matrix
        Input quantum object or csr_matrix.

    Returns
    -------
    degree : array
        Array of integers giving the degree for each node (row).

    """
    if not (sp.isspmatrix_csc(A) or sp.isspmatrix_csr(A)):
        raise TypeError('Input must be CSC or CSR sparse matrix.')
    return _node_degrees(A.indices, A.indptr, A.shape[0])


def breadth_first_search(A, start):
    """
    Breadth-First-Search (BFS) of a graph in CSR or CSC matrix format starting
    from a given node (row).  Takes Qobjs and CSR or CSC matrices as inputs.

    This function requires a matrix with symmetric structure.
    Use A+trans(A) if original matrix is not symmetric or not sure.

    Parameters
    ----------
    A : csc_matrix, csr_matrix
        Input graph in CSC or CSR matrix format
    start : int
        Staring node for BFS traversal.

    Returns
    -------
    order : array
        Order in which nodes are traversed from starting node.
    levels : array
        Level of the nodes in the order that they are traversed.

    """
    if not (sp.isspmatrix_csc(A) or sp.isspmatrix_csr(A)):
        raise TypeError('Input must be CSC or CSR sparse matrix.')

    num_rows = A.shape[0]
    start = int(start)
    order, levels = _breadth_first_search(A.indices, A.indptr, num_rows, start)
    # since maybe not all nodes are in search, check for unused entires in
    # arrays
    return order[order != -1], levels[levels != -1]


def column_permutation(A):
    """
    Finds the non-symmetric column permutation of A such that the columns 
    are given in ascending order according to the number of nonzero entries.
    This is sometimes useful for decreasing the fill-in of sparse LU 
    factorization.
    
    Parameters
    ----------
    A : csc_matrix
        Input sparse CSC sparse matrix.

    Returns
    -------
    perm : array
        Array of permuted row and column indices.
    
    """
    if not sp.isspmatrix_csc(A):
        A = sp.csc_matrix(A)
    count = np.diff(A.indptr)
    perm = np.argsort(count)
    return perm


def reverse_cuthill_mckee(A, sym=False):
    """
    Returns the permutation array that orders a sparse CSR or CSC matrix
    in Reverse-Cuthill McKee ordering. Since the input matrix must be
    symmetric, this routine works on the matrix A+Trans(A) if the sym flag is
    set to False (Default).

    It is assumed by default (*sym=False*) that the input matrix is not
    symmetric. This is because it is faster to do A+Trans(A) than it is to
    check for symmetry for a generic matrix. If you are guaranteed that the
    matrix is symmetric in structure (values of matrix element do not matter)
    then set *sym=True*

    Parameters
    ----------
    A : csc_matrix, csr_matrix
        Input sparse CSC or CSR sparse matrix format.
    sym : bool {False, True}
        Flag to set whether input matrix is symmetric.

    Returns
    -------
    perm : array
        Array of permuted row and column indices.

    Notes
    -----
    This routine is used primarily for internal reordering of Lindblad
    superoperators for use in iterative solver routines.

    References
    ----------
    E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric
    Matrices", ACM '69 Proceedings of the 1969 24th national conference,
    (1969).
    
    """
    if not (sp.isspmatrix_csc(A) or sp.isspmatrix_csr(A)):
        raise TypeError('Input must be CSC or CSR sparse matrix.')

    nrows = A.shape[0]

    if not sym:
        A = A + A.transpose()

    return _reverse_cuthill_mckee(A.indices, A.indptr, nrows)


def maximum_bipartite_matching(A, perm_type='row'):
    """
    Returns an array of row or column permutations that removes nonzero
    elements from the diagonal of a nonsingular square CSC sparse matrix. Such
    a permutation is always possible provided that the matrix is nonsingular.
    This function looks at the structure of the matrix only.

    The input matrix will be converted to CSC matrix format if
    necessary.

    Parameters
    ----------
    A : sparse matrix
        Input matrix

    perm_type : str {'row', 'column'}
        Type of permutation to generate.

    Returns
    -------
    perm : array
        Array of row or column permutations.

    Notes
    -----
    This function relies on a maximum cardinality bipartite matching algorithm
    based on a breadth-first search (BFS) of the underlying graph[1]_.

    References
    ----------
    I. S. Duff, K. Kaya, and B. Ucar, "Design, Implementation, and
    Analysis of Maximum Transversal Algorithms", ACM Trans. Math. Softw.
    38, no. 2, (2011).
    
    """
    nrows = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            'Maximum bipartite matching requires a square matrix.')

    if sp.isspmatrix_csr(A) or sp.isspmatrix_coo(A):
        A = A.tocsc()
    elif not sp.isspmatrix_csc(A):
        raise TypeError("matrix must be in CSC, CSR, or COO format.")

    if perm_type == 'column':
        A = A.transpose().tocsc()

    perm = _maximum_bipartite_matching(A.indices, A.indptr, nrows)

    if np.any(perm == -1):
        raise Exception('Possibly singular input matrix.')

    return perm


def weighted_bipartite_matching(A, perm_type='row'):
    """
    Returns an array of row permutations that attempts to maximize
    the product of the ABS values of the diagonal elements in
    a nonsingular square CSC sparse matrix. Such a permutation is
    always possible provided that the matrix is nonsingular.

    This function looks at both the structure and ABS values of the
    underlying matrix.

    Parameters
    ----------
    A : csc_matrix
        Input matrix

    perm_type : str {'row', 'column'}
        Type of permutation to generate.

    Returns
    -------
    perm : array
        Array of row or column permutations.

    Notes
    -----
    This function uses a weighted maximum cardinality bipartite matching
    algorithm based on breadth-first search (BFS).  The columns are weighted
    according to the element of max ABS value in the associated rows and
    are traversed in descending order by weight.  When performing the BFS
    traversal, the row associated to a given column is the one with maximum
    weight. Unlike other techniques[1]_, this algorithm does not guarantee the
    product of the diagonal is maximized.  However, this limitation is offset
    by the substantially faster runtime of this method.

    References
    ----------
    I. S. Duff and J. Koster, "The design and use of algorithms for
    permuting large entries to the diagonal of sparse matrices", SIAM J.
    Matrix Anal. and Applics. 20, no. 4, 889 (1997).

    """
   
    nrows = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError('weighted_bfs_matching requires a square matrix.')

    if sp.isspmatrix_csr(A) or sp.isspmatrix_coo(A):
        A = A.tocsc()
    elif not sp.isspmatrix_csc(A):
        raise TypeError("matrix must be in CSC, CSR, or COO format.")

    if perm_type == 'column':
        A = A.transpose().tocsc()

    perm = _weighted_bipartite_matching(
                    np.asarray(np.abs(A.data), dtype=float), 
                    A.indices, A.indptr, nrows)

    if np.any(perm == -1):
        raise Exception('Possibly singular input matrix.')

    return perm

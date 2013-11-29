# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
"""
This module contains a collection of graph theory routines used mainly
to reorder matrices for iterative steady state solvers.
"""

import numpy as np
import scipy.sparse as sp
from qutip.cyQ.graph_utils import (_pseudo_peripheral_node, _breadth_first_search,
                                    _node_degrees)
from qutip.settings import debug

if debug:
    import inspect

def graph_degree(A):
    """
    Returns the degree for the nodes (rows) of a graph in
    sparse CSR format.  Takes Qobjs or csr_matrices as inputs.
    
    This function requires a matrix with symmetric structure.
    
    Parameters
    ----------
    A : qobj / csr_matrix
        Input quantum object or csr_matrix.
    
    Returns
    -------
    degree : array
        Array of integers giving the degree for each node (row).
    
    """
    if A.__class__.__name__=='Qobj':
        A=A.data
    return _node_degrees(A.indices, A.indptr, A.shape[0])


def breadth_first_search(A,start):
    """
    Breadth-First-Search (BFS) of a graph in CSR matrix format starting
    from a given node (row).  Takes Qobjs and csr_matrices as inputs.
    
    This function requires a matrix with symmetric structure.
    
    Parameters
    ----------
    A : qobj / csr_matrix
        Input graph in CSR matrix form
    
    start : int
        Staring node for BFS traversal.
    
    Returns
    -------
    order : array
        Order in which nodes are traversed from starting node.
    
    levels : array
        Level of the nodes in the order that they are traversed.
    
    """
    if A.__class__.__name__=='Qobj':
        A=A.data
    num_rows=A.shape[0]
    start=int(start)
    order, levels = _breadth_first_search(A.indices,A.indptr, num_rows, start)
    #since maybe not all nodes are in search, check for unused entires in arrays
    return order[order!=-1], levels[levels!=-1]


def symrcm(A):
    """
    Symmetric Reverse Cuthill-McKee ordering of a Qobj
    or sparse csr_matrix.  Input Qobj/matrix must have a 
    symmetric structure.  
    
    Use A+trans(A) if Qobj/matrix is not symmetric.
    
    This method first attempts to find a pseudo peripheral
    node, and then does a breadth level search.  Because
    the initial seed node is picked at random, the final
    sorting is in general not unique.
    
    Parameters
    ----------
    A : qobj / csr_matrix
    
    Returns
    -------
    perm : array
        Permuted matrix indices.
    
    References
    ----------
    E. Cuthill and J. McKee,
    Reducing the Bandwidth of Sparse Symmetric Matrices, 
    ACM '69 Proceedings of the 1969 24th national conference.
    
    """
    if A.__class__.__name__=='Qobj':
        A=A.data
    else:
         if not sp.isspmatrix_csr(A):
             raise TypeError('Input must be sparse matrix in CSR format.')
    N = A.shape[0]
    root, order, level = _pseudo_peripheral_node(A.indices,A.indptr,N)
    perm = np.argsort(level)
    return perm[::-1]
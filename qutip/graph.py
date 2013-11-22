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

import scipy.sparse as sp
import numpy as np
from qutip.cyQ.graph_utils import _sparse_adjacency_degree, _rcm_ordering
from qutip.settings import debug

if debug:
    import inspect


def sparse_adjacency_degree(A):
    """
    Finds the adjacency elements and associated degree for the nodes (rows)
    of an obj or sparse csr_matrix.
    
    Parameters
    ----------
    A : qobj/csr_matrix
        Input qobj or csr_matrix.
    
    Returns
    -------
    adj : array 
        Adjacency elements for each row (node).
    
    deg : array
        Degree of each row (node).
    
    """
    nrows=A.shape[0]
    if A.__class__.__name__=='Qobj':
        A=A.data
    
    adj, deg = _sparse_adjacency_degree(A.indices,A.indptr,nrows)
    return adj, deg


def reverse_cuthill_mckee_ordering(A,symmetric=False):
    """
    Reverse Cuthill-McKee (RCM) ordering of a sparse csr_matrix.
    Returns the permutation array that reduces the bandwidth
    of the matrix.  Here we pick the node (row) with the
    lowest degree as the starting point. This uses a Queue 
    based method.
    
    If the input matrix is not symmetric (as usual), then the 
    ordering is calculated using A+trans(A).
    
    Parameters
    ----------
    A : csr_matrix
        Sparse csr_matrix for reordering
    
    symmetric : bool
        Flag set if input matrix is symmetric
    
    Returns
    -------
    perm : array
        Permutation array reordering of rows and cols
    
    """
    if not symmetric:
        A=A+A.transpose() #could be made faster since dont need data
    return _rcm_ordering(A.indices,A.indptr,A.shape[0])
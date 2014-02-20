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
import numpy as np
import scipy.sparse as sp
import unittest
from numpy.testing import assert_, run_module_suite, assert_equal
from numpy.testing.decorators import skipif
from scipy.sparse.csgraph import breadth_first_order as BFO
import os
pwd = os.path.dirname(__file__)
from qutip import *
# find networkx if it exists
try:
    import networkx as nx
except:
    nx_found = 0
else:
    nx_found = 1

@unittest.skipIf(nx_found == 0, 'Networkx not installed.')
def test_graph_degree():
    "Graph: Graph Degree"
    A = rand_dm(25,0.5)
    deg = graph_degree(A)
    G = nx.from_scipy_sparse_matrix(A.data)
    nx_deg = G.degree()
    nx_deg = array([nx_deg[k] for k in range(25)])
    assert_equal((deg - nx_deg).all(), 0)

def test_graph_bfs():
    "Graph: Breadth-First Search"
    A = rand_dm(25,0.5)
    A = A.data
    A.data=np.real(A.data)
    seed=np.random.randint(24)
    arr1 = BFO(A,seed)[0]
    arr2 = breadth_first_search(A,seed)[0]
    assert_equal((arr1 - arr2).all(), 0)

def test_graph_rcm():
    "Graph: Reverse Cuthill-McKee Ordering"
    B=np.load(pwd+'/bucky.npy')
    B=sp.csr_matrix(B,dtype=float)
    perm=symrcm(B)
    ans=np.load(pwd+'/bucky_perm.npy')
    assert_equal((perm - ans).all(), 0)


if __name__ == "__main__":
    run_module_suite()

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
import numpy as np
import unittest
from numpy.testing import assert_, run_module_suite, assert_equal
from numpy.testing.decorators import skipif
from scipy.sparse.csgraph import breadth_first_order as BFO
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



if __name__ == "__main__":
    run_module_suite()

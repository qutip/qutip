#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from numpy.testing import assert_, assert_equal, run_module_suite

from qutip import *

class TestRand:
    """
    A test class for the built-in random quantum object generators.
    """

    def testRandUnitary(self):
        "random Unitary"

        U=array([rand_unitary(5) for k in range(5)])
        for k in range(5):
            assert_equal(U[k]*U[k].dag()==qeye(5), True)
    
    def testRandherm(self):
        "random hermitian"

        H=array([rand_herm(5) for k in range(5)])
        for k in range(5):
            assert_equal(H[k].isherm==True, True)
            
    def testRanddm(self):
        "random density matrix"

        R=array([rand_dm(5) for k in range(5)])
        for k in range(5):
            assert_equal(sum(R[k].tr())-1.0<1e-15, True)
            #verify all eigvals are >=0
            assert_(not any(sp_eigs(R[k],vecs=False))<0)
            #verify hermitian
            assert_(R[k].isherm)

    def testRandket(self):
        "random ket"
        P=array([rand_ket(5) for k in range(5)])
        for k in range(5):
            assert_equal(P[k].type=='ket', True)
        
if __name__ == "__main__":
    run_module_suite()
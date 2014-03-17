# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:23:46 2013

@author: dcriger
"""

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

from numpy import abs
from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite, assert_raises
import scipy

from qutip.qobj import Qobj
from qutip.states import basis
from qutip.operators import create, destroy, jmat, identity
from qutip.propagator import propagator
from qutip.random_objects import rand_herm
from qutip.superop_reps import (super_to_choi, choi_to_kraus,
                                choi_to_super, kraus_to_choi,
                                to_super, to_choi, to_kraus)


class TestSuperopReps(object):
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """
    
    def rand_super(self):
        h_5 = rand_herm(5)
        return propagator(h_5, scipy.rand(), [
            create(5), destroy(5), jmat(2, 'z')
        ])

    def test_SuperChoiSuper(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """
        superoperator = self.rand_super()
                               
        choi_matrix = to_choi(superoperator)
        test_supe = to_super(choi_matrix)
        
        # Assert both that the result is close to expected, and has the right
        # type.
        assert_((test_supe - superoperator).norm() < 1e-12)
        assert_(choi_matrix.type == "super" and choi_matrix.superrep == "choi")
        assert_(test_supe.type == "super" and test_supe.superrep == "super")

    def test_ChoiKrausChoi(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """
        superoperator = self.rand_super()
        choi_matrix = to_choi(superoperator)
        kraus_ops = to_kraus(choi_matrix)
        test_choi = kraus_to_choi(kraus_ops)
        
        # Assert both that the result is close to expected, and has the right
        # type.
        assert_((test_choi - choi_matrix).norm() < 1e-12)
        assert_(choi_matrix.type == "super" and choi_matrix.superrep == "choi")
        assert_(test_choi.type == "super" and test_choi.superrep == "choi")
        
    def test_SuperPreservesSelf(self):
        """
        Superoperator: Test that to_super(q) returns q if q is already a
        supermatrix.
        """
        superop = self.rand_super()
        assert_(superop is to_super(superop))
        
    def test_ChoiPreservesSelf(self):
        """
        Superoperator: Test that to_choi(q) returns q if q is already Choi.
        """
        superop = self.rand_super()
        choi = to_choi(superop)
        assert_(choi is to_choi(choi))
        
    def test_iscptp(self):
        """
        Superoperator: Checks a few common usecases for Qobj.iscptp.
        """
        superop = self.rand_super()
        assert_(superop.iscptp) 
        assert_(identity(2).iscptp) # Check that unitaries work, too.
        assert_(not (basis(2).iscptp))
        
    def test_choi_tr(self):
        """
        Superoperator: Checks that the trace of matrices returned by to_choi
        matches that asserted by the docstring for that function.
        """
        for dims in xrange(2, 5):
            assert_(abs(to_choi(identity(dims)).tr() - dims) <= 1e-12)

if __name__ == "__main__":
    run_module_suite()
    

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
from __future__ import division

from numpy import abs
from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite, assert_raises
import scipy

from qutip.qobj import Qobj
from qutip.states import basis
from qutip.operators import create, destroy, jmat, identity, sigmax
from qutip.qip.gates import swap
from qutip.propagator import propagator
from qutip.random_objects import rand_herm, rand_super
from qutip.tensor import tensor, super_tensor
from qutip.superop_reps import (super_to_choi, choi_to_kraus,
                                choi_to_super, kraus_to_choi,
                                to_super, to_choi, to_kraus, to_chi,
                                _dep_choi)

tol = 1e-10


class TestSuperopReps(object):
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """

    def test_SuperChoiSuper(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """
        superoperator = rand_super()

        choi_matrix = to_choi(superoperator)
        test_supe = to_super(choi_matrix)

        # Assert both that the result is close to expected, and has the right
        # type.
        assert_((test_supe - superoperator).norm() < tol)
        assert_(choi_matrix.type == "super" and choi_matrix.superrep == "choi")
        assert_(test_supe.type == "super" and test_supe.superrep == "super")

    def test_SuperChoiChiSuper(self):
        """
        Superoperator: Converting two-qubit superoperator through
        Choi and chi representations goes back to right superoperator.
        """
        superoperator = super_tensor(rand_super(2), rand_super(2))

        choi_matrix = to_choi(superoperator)
        chi_matrix = to_chi(choi_matrix)
        test_supe = to_super(chi_matrix)

        # Assert both that the result is close to expected, and has the right
        # type.
        assert_((test_supe - superoperator).norm() < tol)
        assert_(choi_matrix.type == "super" and choi_matrix.superrep == "choi")
        assert_(chi_matrix.type == "super" and chi_matrix.superrep == "chi")
        assert_(test_supe.type == "super" and test_supe.superrep == "super")

    def test_ChoiKrausChoi(self):
        """
        Superoperator: Convert superoperator to Choi matrix and back.
        """
        superoperator = rand_super()
        choi_matrix = to_choi(superoperator)
        kraus_ops = to_kraus(choi_matrix)
        test_choi = kraus_to_choi(kraus_ops)

        # Assert both that the result is close to expected, and has the right
        # type.
        assert_((test_choi - choi_matrix).norm() < tol)
        assert_(choi_matrix.type == "super" and choi_matrix.superrep == "choi")
        assert_(test_choi.type == "super" and test_choi.superrep == "choi")

    def test_SuperPreservesSelf(self):
        """
        Superoperator: to_super(q) returns q if q is already a
        supermatrix.
        """
        superop = rand_super()
        assert_(superop is to_super(superop))

    def test_ChoiPreservesSelf(self):
        """
        Superoperator: to_choi(q) returns q if q is already Choi.
        """
        superop = rand_super()
        choi = to_choi(superop)
        assert_(choi is to_choi(choi))

    def test_random_iscptp(self):
        """
        Superoperator: Randomly generated superoperators are
        correctly reported as cptp.
        """
        superop = rand_super()
        assert_(superop.iscptp)

    def test_known_iscptp(self):
        """
        Superoperator: iscp, istp and iscptp known cases.
        """
        # Check that unitaries are CPTP.
        assert_(identity(2).iscptp)
        assert_(sigmax().iscptp)

        # The partial transpose map, whose Choi matrix is SWAP, is TP but not
        # CP.
        W = Qobj(swap(), type='super', superrep='choi')
        assert_(W.istp)
        assert_(not W.iscp)
        assert_(not W.iscptp)

        # Subnormalized maps (representing erasure channels, for instance)
        # can be CP but not TP.
        subnorm_map = Qobj(identity(4) * 0.9, type='super', superrep='super')
        assert_(subnorm_map.iscp)
        assert_(not subnorm_map.istp)
        assert_(not subnorm_map.iscptp)

        # Check that things which aren't even operators aren't identified as
        # CPTP.
        assert_(not (basis(2).iscptp))

    def test_choi_tr(self):
        """
        Superoperator: Trace returned by to_choi matches docstring.
        """
        for dims in range(2, 5):
            assert_(abs(to_choi(identity(dims)).tr() - dims) <= tol)

if __name__ == "__main__":
    run_module_suite()

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

from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite
import scipy

from qutip.qobj import Qobj
from qutip.operators import create, destroy, jmat
from qutip.propagator import propagator
from qutip.random_objects import rand_herm
from qutip.superop_reps import (super_to_choi, choi_to_kraus,
                                choi_to_super, kraus_to_choi)


class TestSuperopReps(object):
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """

    def test_SuperChoiSuper(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """
        h_5 = rand_herm(5)
        superoperator = propagator(h_5, scipy.rand(),
                                   [create(5), destroy(5), jmat(2, 'z')])
        choi_matrix = super_to_choi(superoperator)
        test_supe = choi_to_super(choi_matrix)
        assert_((test_supe - superoperator).norm() < 1e-12)

    def test_ChoiKrausChoi(self):
        """
        Superoperator: Converting superoperator to Choi matrix and back.
        """
        h_5 = rand_herm(5)
        superoperator = propagator(h_5, scipy.rand(),
                                   [create(5), destroy(5), jmat(2, 'z')])
        choi_matrix = super_to_choi(superoperator)
        kraus_ops = choi_to_kraus(choi_matrix)
        test_choi = kraus_to_choi(kraus_ops)
        assert_((test_choi - choi_matrix).norm() < 1e-12)


if __name__ == "__main__":
    run_module_suite()

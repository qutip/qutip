# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
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

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth
# @date: Sep 2015

"""
Tests for main control.pulseoptim methods
"""
from __future__ import division

from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal)

from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform

import qutip.control.pulseoptim as cpo

class TestPulseOptim:
    """
    A test class for the QuTiP functions for generating quantum gates
    """
    
    def test_unitary(self):
        """
        Optimise pulse for Hadamard and QFT gate with linear initial pulses
        assert that goal is achieved and fidelity error is below threshold
        """
        H_d = sigmaz()
        H_c = [sigmax()]
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        n_ts = 10
        evo_time = 6
        
        # Run the optimisation
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
                        
        assert_(result.goal_achieved)
        assert_almost_equal(result.fid_err, 0.0, decimal=10)
    
#    def test_lindbladian(self):
#        
#    
#    def test_symplectic(self):
    
    
if __name__ == "__main__":
    run_module_suite()
    
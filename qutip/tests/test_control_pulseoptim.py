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

import numpy as np
from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal)
from scipy.optimize import check_grad

from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.qip import hadamard_transform
from qutip.qip.algorithms import qft

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
        # Hadamard
        H_d = sigmaz()
        H_c = [sigmax()]
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        n_ts = 10
        evo_time = 6
        
        # Run the optimisation
        result = cpo.optimize_pulse_unitary(H_d, list(H_c), U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
                        
        assert_(result.goal_achieved, msg="Hadamard goal not achieved")
        assert_almost_equal(result.fid_err, 0.0, decimal=10, 
                            err_msg="Hadamard infidelity too high")
                            
        # Check same result is achieved using the create objects method
        optim = cpo.create_pulse_optimizer(H_d, list(H_c), U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        dyn_type='UNIT', 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        dyn = optim.dynamics

        init_amps = optim.pulse_generator.gen_pulse().reshape([-1, 1])
        dyn.initialize_controls(init_amps)

        # Check the exact gradient
        func = optim.fid_err_func_wrapper
        grad = optim.fid_err_grad_wrapper
        x0 = dyn.ctrl_amps.flatten()
        grad_diff = check_grad(func, grad, x0)
        assert_almost_equal(grad_diff, 0.0, decimal=7,
                            err_msg="Unitary gradient outside tolerance")

        result2 = optim.run_optimization()
        assert_almost_equal(result.fid_err, result2.fid_err, decimal=10, 
                            err_msg="Direct and indirect methods produce "
                                    "different results for Hadamard")
                                    
        # QFT
        Sx = sigmax()
        Sy = sigmay()
        Sz = sigmaz()
        Si = 0.5*identity(2)
        
        H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
        H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
        #n_ctrls = len(H_c)
        U_0 = identity(4)
        # Target for the gate evolution - Quantum Fourier Transform gate
        U_targ = qft.qft(2)
        result = cpo.optimize_pulse_unitary(H_d, list(H_c), U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-9, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
                        
        assert_(result.goal_achieved, msg="QFT goal not achieved")
        assert_almost_equal(result.fid_err, 0.0, decimal=7, 
                            err_msg="QFT infidelity too high")
                            
        # check bounds
        result2 = cpo.optimize_pulse_unitary(H_d, list(H_c), U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-9, 
                        amp_lbound=-1.0, amp_ubound=1.0,
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_((result2.final_amps >= -1.0).all() and 
                    (result2.final_amps <= 1.0).all(), 
                    msg="Amplitude bounds exceeded for QFT")
    
    def test_lindbladian(self):
        """
        Optimise pulse for amplitude damping channel with Lindbladian dyn
        assert that fidelity error is below threshold
        """

        Sx = sigmax()
        Sz = sigmaz()
        Si = identity(2)
        
        Sd = Qobj(np.array([[0, 1], [0, 0]]))
        Sm = Qobj(np.array([[0, 0], [1, 0]]))
        Sd_m = Qobj(np.array([[1, 0], [0, 0]]))
        
        gamma = 0.1
        L0_Ad = gamma*(2*tensor(Sm, Sd.trans()) - 
                    (tensor(Sd_m, Si) + tensor(Si, Sd_m.trans())))
        LC_x = -1j*(tensor(Sx, Si) - tensor(Si, Sx))
        LC_z = -1j*(tensor(Sz, Si) - tensor(Si, Sz))
        
        drift = L0_Ad
        ctrls = [LC_z, LC_x]
        n_ctrls = len(ctrls)
        initial = identity(4)
        had_gate = hadamard_transform(1)
        target_DP = tensor(had_gate, had_gate)

        n_ts = 10
        evo_time = 5
        
        result = cpo.optimize_pulse(drift, list(ctrls), initial, target_DP, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
                        max_iter=200,
                        init_pulse_type='LIN', 
                        gen_stats=True)
       
        assert_(result.fid_err < 0.1, 
                msg="Fidelity higher than expected")
                
        # Check same result is achieved using the create objects method
        optim = cpo.create_pulse_optimizer(drift, list(ctrls), 
                        initial, target_DP,
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        dyn = optim.dynamics

        p_gen = optim.pulse_generator
        init_amps = np.zeros([n_ts, n_ctrls])
        for j in range(n_ctrls):
            init_amps[:, j] = p_gen.gen_pulse()
        dyn.initialize_controls(init_amps)

        # Check the exact gradient
        func = optim.fid_err_func_wrapper
        grad = optim.fid_err_grad_wrapper
        x0 = dyn.ctrl_amps.flatten()
        grad_diff = check_grad(func, grad, x0)
        assert_almost_equal(grad_diff, 0.0, decimal=7,
                            err_msg="Frechet gradient outside tolerance")

        result2 = optim.run_optimization()
        assert_almost_equal(result.fid_err, result2.fid_err, decimal=3, 
                            err_msg="Direct and indirect methods produce "
                                    "different results for ADC")

#    
#    def test_symplectic(self):
    
    
if __name__ == "__main__":
    run_module_suite()
    
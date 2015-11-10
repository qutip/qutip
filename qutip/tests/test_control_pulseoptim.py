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

import os
import numpy as np
from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal)
from scipy.optimize import check_grad

from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.qip import hadamard_transform
from qutip.qip.algorithms import qft
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
import qutip.control.pulsegen as pulsegen
import qutip.control.errors as errors
import qutip.control.loadparams as loadparams
import qutip.control.pulseoptim as cpo
import qutip.control.symplectic as sympl

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
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved")
        assert_almost_equal(result.fid_err, 0.0, decimal=10, 
                            err_msg="Hadamard infidelity too high")
                            
        #Try without stats
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=False)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved "
                                            "(no stats)")
                                            
        #Try with Qobj propagation
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        dyn_params={'oper_dtype':Qobj},
                        gen_stats=True)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved "
                                            "(Qobj propagation)")
        
        # Check same result is achieved using the create objects method
        optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, 
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
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-9, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
                        
        assert_(result.goal_achieved, msg="QFT goal not achieved")
        assert_almost_equal(result.fid_err, 0.0, decimal=7, 
                            err_msg="QFT infidelity too high")
                            
        # check bounds
        result2 = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
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
        initial = tensor(Si, Si)
        had_gate = hadamard_transform(1)
        target_DP = tensor(had_gate, had_gate)

        n_ts = 10
        evo_time = 5
        
        result = cpo.optimize_pulse(drift, ctrls, initial, target_DP, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
                        max_iter=200,
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result.fid_err < 0.1, 
                msg="Fidelity higher than expected")
                
        # Repeat with Qobj propagation
        result = cpo.optimize_pulse(drift, ctrls, initial, target_DP, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
                        max_iter=200,
                        init_pulse_type='LIN', 
                        dyn_params={'oper_dtype':Qobj},
                        gen_stats=True)
        assert_(result.fid_err < 0.1, 
                msg="Fidelity higher than expected (Qobj propagation)")
                
        # Check same result is achieved using the create objects method
        optim = cpo.create_pulse_optimizer(drift, ctrls, 
                        initial, target_DP,
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
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

    def test_symplectic(self):
        """
        Optimise pulse for coupled oscillators with Symplectic dynamics
        assert that fidelity error is below threshold
        """
        g1 = 1.0
        g2 = 0.2
        A0 = Qobj(np.array([[1, 0, g1, 0], 
                           [0, 1, 0, g2], 
                           [g1, 0, 1, 0], 
                           [0, g2, 0, 1]]))
        A_rot = Qobj(np.array([
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]
                            ]))
        
        A_sqz = Qobj(0.4*np.array([
                            [1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]
                            ]))
                            
        A_c = [A_rot, A_sqz]
        n_ctrls = len(A_c)
        initial = identity(4)
        A_targ = Qobj(np.array([
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1], 
                        [1, 0, 0, 0], 
                        [0, 1, 0, 0]
                        ]))
        Omg = Qobj(sympl.calc_omega(2))
        S_targ = (-A_targ*Omg*np.pi/2.0).expm()
    
        n_ts = 20
        evo_time = 10
        
        result = cpo.optimize_pulse(A0, A_c, initial, S_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
                        max_iter=200,
                        dyn_type='SYMPL',
                        init_pulse_type='ZERO', 
                        gen_stats=True)
        assert_(result.goal_achieved, msg="Symplectic goal not achieved")
        assert_almost_equal(result.fid_err, 0.0, decimal=2, 
                            err_msg="Symplectic infidelity too high")
        
        # Repeat with Qobj integration
        resultq = cpo.optimize_pulse(A0, A_c, initial, S_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
                        max_iter=200,
                        dyn_type='SYMPL',
                        init_pulse_type='ZERO', 
                        dyn_params={'oper_dtype':Qobj},
                        gen_stats=True)
        assert_(result.goal_achieved, msg="Symplectic goal not achieved "
                                        "(Qobj integration)")
        
        # Check same result is achieved using the create objects method
        optim = cpo.create_pulse_optimizer(A0, list(A_c), 
                        initial, S_targ,
                        n_ts, evo_time, 
                        fid_err_targ=1e-3, 
                        dyn_type='SYMPL',
                        init_pulse_type='ZERO', 
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
        assert_almost_equal(grad_diff, 0.0, decimal=5,
                            err_msg="Frechet gradient outside tolerance "
                                    "(SYMPL)")

        result2 = optim.run_optimization()
        assert_almost_equal(result.fid_err, result2.fid_err, decimal=6, 
                            err_msg="Direct and indirect methods produce "
                                    "different results for Symplectic")
                                    
    def test_crab(self):
        """
        Optimise pulse for Hadamard gate using CRAB algorithm
        Apply guess and ramping pulse
        assert that goal is achieved and fidelity error is below threshold
        assert that starting amplitude is zero
        """
        # Hadamard
        H_d = sigmaz()
        H_c = [sigmax()]
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        n_ts = 12
        evo_time = 10
        
        # Run the optimisation
        result = cpo.opt_pulse_crab_unitary(H_d, H_c, U_0, U_targ, 
                n_ts, evo_time, 
                fid_err_targ=1e-5, 
                alg_params={'crab_pulse_params':{'randomize_coeffs':False, 
                                                 'randomize_freqs':False}},
                init_coeff_scaling=0.5,
                guess_pulse_type='GAUSSIAN', 
                guess_pulse_params={'variance':0.1*evo_time},
                guess_pulse_scaling=1.0, guess_pulse_offset=1.0,
                amp_lbound=None, amp_ubound=None,
                ramping_pulse_type='GAUSSIAN_EDGE', 
                ramping_pulse_params={'decay_time':evo_time/100.0},
                gen_stats=True)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved")
        assert_almost_equal(result.fid_err, 0.0, decimal=3, 
                            err_msg="Hadamard infidelity too high")
        assert_almost_equal(result.final_amps[0, 0], 0.0, decimal=3, 
                            err_msg="lead in amplitude not zero")
        # Repeat with Qobj integration
        result = cpo.opt_pulse_crab_unitary(H_d, H_c, U_0, U_targ, 
                n_ts, evo_time, 
                fid_err_targ=1e-5, 
                alg_params={'crab_pulse_params':{'randomize_coeffs':False, 
                                                 'randomize_freqs':False}},
                dyn_params={'oper_dtype':Qobj},
                init_coeff_scaling=0.5,
                guess_pulse_type='GAUSSIAN', 
                guess_pulse_params={'variance':0.1*evo_time},
                guess_pulse_scaling=1.0, guess_pulse_offset=1.0,
                amp_lbound=None, amp_ubound=None,
                ramping_pulse_type='GAUSSIAN_EDGE', 
                ramping_pulse_params={'decay_time':evo_time/100.0},
                gen_stats=True)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved" 
                                        "(Qobj integration)")
                            
    def test_load_params(self):
        """
        Optimise pulse for Hadamard gate by loading config from file
        compare with result produced by pulseoptim method
        """
        H_d = sigmaz()
        H_c = sigmax()
        
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        cfg = optimconfig.OptimConfig()
        cfg.param_fname = "Hadamard_params.ini"
        cfg.param_fpath = os.path.join(os.path.dirname(__file__), 
                                           cfg.param_fname)
        cfg.pulse_type = "ZERO"
        loadparams.load_parameters(cfg.param_fpath, config=cfg)

        dyn = dynamics.DynamicsUnitary(cfg)
        dyn.target = U_targ
        dyn.initial = U_0
        dyn.drift_dyn_gen = H_d
        dyn.ctrl_dyn_gen = [H_c]
        loadparams.load_parameters(cfg.param_fpath, dynamics=dyn)
        dyn.init_timeslots()      
        n_ts = dyn.num_tslots
        n_ctrls = dyn.get_num_ctrls()
        
        pgen = pulsegen.create_pulse_gen(pulse_type=cfg.pulse_type, dyn=dyn)
        loadparams.load_parameters(cfg.param_fpath, pulsegen=pgen)
        
        tc = termcond.TerminationConditions()
        loadparams.load_parameters(cfg.param_fpath, term_conds=tc)
        
        if cfg.optim_method == 'BFGS':
            optim = optimizer.OptimizerBFGS(cfg, dyn)
        elif cfg.optim_method == 'FMIN_L_BFGS_B':
            optim = optimizer.OptimizerLBFGSB(cfg, dyn)
        elif cfg.optim_method is None:
            raise errors.UsageError("Optimisation algorithm must be specified "
                                    "via 'optim_method' parameter")
        else:
            optim = optimizer.Optimizer(cfg, dyn)
            optim.method = cfg.optim_method
        loadparams.load_parameters(cfg.param_fpath, optim=optim)
        
        sts = stats.Stats()
        dyn.stats = sts
        optim.stats = sts
        optim.config = cfg
        optim.dynamics = dyn
        optim.pulse_generator = pgen
        optim.termination_conditions = tc
        
        init_amps = np.zeros([n_ts, n_ctrls])
        for j in range(n_ctrls):
            init_amps[:, j] = pgen.gen_pulse()
        dyn.initialize_controls(init_amps)
        result = optim.run_optimization()
        
        result2 = cpo.optimize_pulse_unitary(H_d, list([H_c]), U_0, U_targ, 
                        6, 6, fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        amp_lbound=-1.0, amp_ubound=1.0,
                        gen_stats=True)
                        
        assert_almost_equal(result.final_amps, result2.final_amps, decimal=5, 
                            err_msg="Pulses do not match")

if __name__ == "__main__":
    run_module_suite()
    
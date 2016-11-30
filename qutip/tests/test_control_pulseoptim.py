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
Some associated objects also tested.
"""
from __future__ import division

import os
import uuid
import shutil
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
    
    def setUp(self):
        # list of file paths to be removed after test
        self.tmp_files = []
        # list of folder paths to be removed after test
        self.tmp_dirs = [] 
    
    def tearDown(self):
        for f in self.tmp_files:
            try:
                os.remove(f)
            except:
                pass
            
        for d in self.tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
            
            
    def test_1_unitary(self):
        """
        control.pulseoptim: Hadamard and QFT gate with linear initial pulses
        assert that goal is achieved and fidelity error is below threshold
        """
        # Hadamard
        H_d = sigmaz()
        H_c = [sigmax()]
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        n_ts = 10
        evo_time = 10
        
        # Run the optimisation
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
        assert_almost_equal(result.fid_err, 0.0, decimal=10, 
                            err_msg="Hadamard infidelity too high")
                            
        #Try without stats
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=False)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved "
                                            "(no stats). "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
                    
        #Try setting timeslots with tau array
        tau = np.arange(1.0, 10.0, 1.0)
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        tau=tau, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=False)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved "
                                            "(tau as timeslots). "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
                                            
        #Try with Qobj propagation
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        dyn_params={'oper_dtype':Qobj},
                        gen_stats=True)
        assert_(result.goal_achieved, msg="Hadamard goal not achieved "
                                            "(Qobj propagation). "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
        
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
                        
        assert_(result.goal_achieved, msg="QFT goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
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
                    
    def test_2_dumping_and_unitarity(self):
        """
        control: data dumping and unitarity checking
        Dump out processing data and use to check unitary evolution
        """
        N_EXP_OPTIMDUMP_FILES = 10
        N_EXP_DYNDUMP_FILES = 49
        
        # Hadamard
        H_d = sigmaz()
        H_c = [sigmax()]
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        n_ts = 1000
        evo_time = 4
        
        dump_folder = str(uuid.uuid4())
        qtrl_dump_dir = os.path.expanduser(os.path.join('~', dump_folder))
        self.tmp_dirs.append(qtrl_dump_dir)
        optim_dump_dir = os.path.join(qtrl_dump_dir, 'optim')
        dyn_dump_dir = os.path.join(qtrl_dump_dir, 'dyn')
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-9, 
                        init_pulse_type='LIN', 
                        optim_params={'dumping':'FULL', 'dump_to_file':True, 
                                    'dump_dir':optim_dump_dir},
                        dyn_params={'dumping':'FULL', 'dump_to_file':True, 
                                    'dump_dir':dyn_dump_dir},
                        gen_stats=True)
        
        # check dumps were generated
        optim = result.optimizer
        dyn = optim.dynamics
        assert_(optim.dump is not None, msg='optimizer dump not created')
        assert_(dyn.dump is not None, msg='dynamics dump not created')
        
        # Count files that were output
        nfiles = len(os.listdir(optim.dump.dump_dir))
        assert_(nfiles == N_EXP_OPTIMDUMP_FILES, 
                msg="{} optimizer dump files generated, {} expected".format(
                    nfiles, N_EXP_OPTIMDUMP_FILES))
                    
        nfiles = len(os.listdir(dyn.dump.dump_dir))
        assert_(nfiles == N_EXP_DYNDUMP_FILES, 
                msg="{} dynamics dump files generated, {} expected".format(
                    nfiles, N_EXP_DYNDUMP_FILES))
                    
        # dump all to specific file stream
        fpath = os.path.expanduser(os.path.join('~', str(uuid.uuid4())))
        self.tmp_files.append(fpath)
        with open(fpath, 'wb') as f:
            optim.dump.writeout(f)
        
        assert_(os.stat(fpath).st_size > 0, msg="Nothing written to optimizer dump file")
        
        fpath = os.path.expanduser(os.path.join('~', str(uuid.uuid4())))
        self.tmp_files.append(fpath)
        with open(fpath, 'wb') as f:
            dyn.dump.writeout(f)
        assert_(os.stat(fpath).st_size > 0, msg="Nothing written to dynamics dump file")
        
        # Use the dump to check unitarity of all propagators and evo_ops
        dyn.unitarity_tol = 1e-14
        nu_prop = 0
        nu_fwd_evo = 0
        nu_onto_evo = 0
        for d in dyn.dump.evo_dumps:
            for k in range(dyn.num_tslots):
                if not dyn._is_unitary(d.prop[k]): nu_prop += 1
                if not dyn._is_unitary(d.fwd_evo[k]): nu_fwd_evo += 1
                if not dyn._is_unitary(d.onto_evo[k]): nu_onto_evo += 1
        assert_(nu_prop==0, 
                msg="{} propagators found to be non-unitary".format(nu_prop))
        assert_(nu_fwd_evo==0, 
                msg="{} fwd evo ops found to be non-unitary".format(
                                                                nu_fwd_evo))
        assert_(nu_onto_evo==0,
                msg="{} onto evo ops found to be non-unitary".format(
                                                                nu_onto_evo))
            
    def test_3_state_to_state(self):
        """
        control.pulseoptim: state-to-state transfer 
        linear initial pulse used
        assert that goal is achieved
        """       
        # 2 qubits with Ising interaction
        # some arbitary coupling constants
        alpha = [0.9, 0.7]
        beta  = [0.8, 0.9]
        Sx = sigmax()
        Sz = sigmaz()
        H_d = (alpha[0]*tensor(Sx,identity(2)) + 
              alpha[1]*tensor(identity(2),Sx) +
              beta[0]*tensor(Sz,identity(2)) +
              beta[1]*tensor(identity(2),Sz))
        H_c = [tensor(Sz,Sz)]
        
        q1_0 = q2_0 = Qobj([[1], [0]])
        q1_T = q2_T = Qobj([[0], [1]])
        
        psi_0 = tensor(q1_0, q2_0)
        psi_T = tensor(q1_T, q2_T)
        
        n_ts = 10
        evo_time = 18
        
        # Run the optimisation
        result = cpo.optimize_pulse_unitary(H_d, H_c, psi_0, psi_T, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result.goal_achieved, msg="State-to-state goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
        assert_almost_equal(result.fid_err, 0.0, decimal=10, 
                            err_msg="Hadamard infidelity too high")
                            
        #Try with Qobj propagation
        result = cpo.optimize_pulse_unitary(H_d, H_c, psi_0, psi_T, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        dyn_params={'oper_dtype':Qobj},
                        gen_stats=True)
        assert_(result.goal_achieved, msg="State-to-state goal not achieved "
                    "(Qobj propagation)"
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
                                            
    def test_4_lindbladian(self):
        """
        control.pulseoptim: amplitude damping channel
        Lindbladian dynamics
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

    def test_5_symplectic(self):
        """
        control.pulseoptim: coupled oscillators (symplectic dynamics)
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
        assert_(result.goal_achieved, msg="Symplectic goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
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
        assert_(resultq.goal_achieved, msg="Symplectic goal not achieved "
                                        "(Qobj integration). "
                    "Terminated due to: {}, with infidelity: {}".format(
                    resultq.termination_reason, result.fid_err))
                    
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
                                    
    def test_6_crab(self):
        """
        control.pulseoptim: Hadamard gate using CRAB algorithm
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
        assert_(result.goal_achieved, msg="Hadamard goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
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
                                        "(Qobj integration). "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result.termination_reason, result.fid_err))
                    
    def test_7_load_params(self):
        """
        control.pulseoptim: Hadamard gate (loading config from file)
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
        n_ctrls = dyn.num_ctrls
        
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
                            
    
    def test_8_init_pulse_params(self):
        """
        control.pulsegen: Check periodic control functions
        """
        
        def count_waves(n_ts, evo_time, ptype, freq=None, num_waves=None):
            
            # Any dyn config will do 
            #Hadamard
            H_d = sigmaz()
            H_c = [sigmax()]
            U_0 = identity(2)
            U_targ = hadamard_transform(1)
            
            pulse_params = {}
            if freq is not None:
                pulse_params['freq'] = freq
            if num_waves is not None:
                pulse_params['num_waves'] = num_waves
            
            optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, 
                                        n_ts, evo_time, 
                                        dyn_type='UNIT', 
                                        init_pulse_type=ptype,
                                        init_pulse_params=pulse_params,
                                        gen_stats=False)
            pgen = optim.pulse_generator
            pulse = pgen.gen_pulse()
            
            # count number of waves
            zero_cross = pulse[0:-2]*pulse[1:-1] < 0
            
            return (sum(zero_cross) + 1) / 2
        
        n_ts = 1000
        evo_time = 10
        
        ptypes = ['SINE', 'SQUARE', 'TRIANGLE', 'SAW']
        numws = [1, 5, 10, 100]
        freqs = [0.1, 1, 10, 20]
        
        for ptype in ptypes:
            for freq in freqs:
                exp_num_waves = evo_time*freq
                fnd_num_waves = count_waves(n_ts, evo_time, ptype, freq=freq)
#                print("Found {} waves for pulse type '{}', "
#                    "freq {}".format(fnd_num_waves, ptype, freq))
                assert_equal(exp_num_waves, fnd_num_waves, err_msg=
                    "Number of waves incorrect for pulse type '{}', "
                    "freq {}".format(ptype, freq))
                    
            for num_waves in numws:
                exp_num_waves = num_waves
                fnd_num_waves = count_waves(n_ts, evo_time, ptype, 
                                            num_waves=num_waves)
#                print("Found {} waves for pulse type '{}', "
#                    "num_waves {}".format(fnd_num_waves, ptype, num_waves))
                assert_equal(exp_num_waves, fnd_num_waves, err_msg=
                    "Number of waves incorrect for pulse type '{}', "
                    "num_waves {}".format(ptype, num_waves))
                    
    def test_9_time_dependent_drift(self):
        """
        control.pulseoptim: Hadamard gate with fixed and time varying drift
        assert that goal is achieved for both and that different control
        pulses are produced (only) when they should be
        """
        # Hadamard
        H_0 = sigmaz()
        H_c = [sigmax()]
        U_0 = identity(2)
        U_targ = hadamard_transform(1)

        n_ts = 20
        evo_time = 10
        
        drift_amps_flat = np.ones([n_ts], dtype=float)
        dript_amps_step = [np.round(float(k)/n_ts) for k in range(n_ts)]
        
        # Run the optimisations
        result_fixed = cpo.optimize_pulse_unitary(H_0, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result_fixed.goal_achieved, 
                    msg="Fixed drift goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result_fixed.termination_reason, result_fixed.fid_err))
                    
        H_d = [drift_amps_flat[k]*H_0 for k in range(n_ts)]
        result_flat = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result_flat.goal_achieved, msg="Flat drift goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result_flat.termination_reason, result_flat.fid_err))
                    
        # Check fixed and flat produced the same pulse
        assert_almost_equal(result_fixed.final_amps, result_flat.final_amps, 
                            decimal=9, 
                            err_msg="Flat and fixed drift result in "
                                    "different control pules")
                            
        H_d = [dript_amps_step[k]*H_0 for k in range(n_ts)]
        result_step = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, 
                        n_ts, evo_time, 
                        fid_err_targ=1e-10, 
                        init_pulse_type='LIN', 
                        gen_stats=True)
        assert_(result_step.goal_achieved, msg="Step drift goal not achieved. "
                    "Terminated due to: {}, with infidelity: {}".format(
                    result_step.termination_reason, result_step.fid_err))
                    
        # Check step and flat produced different results
        assert_(np.any(
            np.abs(result_flat.final_amps - result_step.final_amps) > 1e-3), 
                            msg="Flat and step drift result in "
                                    "the same control pules")
                                    
if __name__ == "__main__":
    run_module_suite()
    
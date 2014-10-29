# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 21:40:04 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Wrapper functions that will manage the creation of the objects,
build the configuration, and execute the algorithm required to optimise
a set of ctrl pulses for a given (quantum) system.
The fidelity error is some measure of distance of the system evolution
from the given target evolution in the time allowed for the evolution.
The functions minimise this fidelity error wrt the piecewise control
amplitudes in the timeslots
"""
import numpy as np

import optimconfig
import dynamics
import termcond
import optimizer
import stats
import errors
import fidcomp
import propcomp
import pulsegen

def optimize_pulse(\
            drift, Ctrls, initial, target, \
            num_tslots=None, evo_time=None, tau=None, \
            amp_lbound=-np.Inf, amp_ubound=np.Inf, \
            fid_err_targ=1e-10, min_grad = 1e-10, \
            max_iter=500, max_wall_time=180, \
            optim_alg='LBFGSB', dyn_type='GEN_MAT', prop_type='DEF', \
            fid_type='DEF', phase_option=None, amp_update_mode='ALL', \
            init_pulse_type='RND', \
            msg_level=0, out_file_ext='.txt', gen_stats=False):
    """
    Optimise a ctrl pulse to minimise the fidelity error.
    The dynamics of the system in any given timeslot are governed 
    by the combined dynamics generator, 
    i.e. the sum of the drift+ctrl_amp[j]*Ctrls[j]
    The ctrl pulse is an [n_ts, len(Ctrls)] array of piecewise amplitudes
    Starting from an intital (typically random) pulse, 
    a multivariable optimisation algorithm attempts to determines the 
    optimal values for the ctrl pulse to minimise the fidelity error
    The fidelity error is some measure of distance of the system evolution
    from the given target evolution in the time allowed for the evolution.
    Parameters:
        drift - (aka system) the underlying dynamics generator of the system
        Ctrls - a list of control dynamics generators. These are scaled by
                the amplitudes to alter the overall dynamics
        initial - starting point for the evolution. 
                Typically the identity matrix
        target - target transformation, e.g. gate or state, 
                for the time evolution
        num_tslots - number of timeslots
        evo_time - total time for the evolution
        tau - array of durations for the timeslots.
                if this is given then num_tslots and evo_time are dervived
                from it
        amp_lbound / amp_ubound - lower and upper boundaries for the ctrl
                ctrl amplitudes. Can be a scalar value applied to all ctrls
                or a list of bounds for each ctrl
        fid_err_targ - Fidelity error target. Pulse optimisation will 
                terminate when the fidelity error falls below this value
        mim_grad - Minimum gradient. When the sum of the squares of the
                gradients wrt to the ctrl amplitudes falls below this
                value, the optimisation terminates, assuming local minima
        max_iter - Maximum number of iterations of the optimisation algorithm
        max_wall_time - Maximum allowed elapsed time for the 
                optimisation algorithm
        optim_alg - Multi-variable optimisation algorithm
                options are BFGS, LBFGSB
                (see Optimizer classes for details)
        dyn_type - Dynamics type, i.e. the type of matrix used to describe
                the dynamics. Options are UNIT, GEN_MAT, SYMPL
                (see Dynamics classes for details)
        prop_type - i.e. the method used to calculate the
                propagtors and propagtor gradient for each timeslot
                options are APPROX, DIAG, FRECHET, AUG_MAT
                (see PropagatorComputer classes for details)
        fid_type - Fidelity error and (fidelity error gradient) computation
                method. Options are UNIT, TRACEDIFF, TD_APPROX
                (See FideliyComputer classes for details)
        phase_option - determines how global phase is treated in fidelity
                calculations (fid_type='UNIT' only). Options:
                    PSU - global phase ignored
                    SU - global phase included
        amp_update_mode - determines whether propagators are calculated
                Options: ALL, DYNAMIC (needs work)
                (See TimeslotComputer classes for details)
        init_pulse_type - type / shape of pulse(s) used to initialise the
                the ctrl amplitudes. Options include:
                    RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
                (see PulseGen classes for details)
        msg_level - level of messaging output to the stdout.
                0 -> no messaging, up to about 5 for lots of debug messages
        out_file_ext - files containing the initial and final ctrl pulse
                amplitudes are saved to the current directory. 
                The default name will be postfixed with this extension
                Setting this to None will suppress the ouput of files
        gen_stats - if set to True then statistics for the optimisation
                run will be generated - accessible through attributes 
                of the stats object
    Returns:
        Returns an OptimResult object which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    """
    optim = create_pulse_optimizer(\
            drift, Ctrls, initial, target, \
            num_tslots=num_tslots, evo_time=evo_time, tau=tau, \
            amp_lbound=amp_lbound, amp_ubound=amp_ubound, \
            fid_err_targ=fid_err_targ, min_grad=min_grad, \
            max_iter=max_iter, max_wall_time=max_wall_time, \
            optim_alg=optim_alg, dyn_type=dyn_type, prop_type=prop_type, \
            fid_type=fid_type, phase_option=phase_option, \
            amp_update_mode=amp_update_mode, init_pulse_type=init_pulse_type, \
            msg_level=msg_level, gen_stats=gen_stats)
    
    dyn = optim.dynamics
    p_gen = optim.pulse_generator
    
    if (msg_level >= 1):
        print "System configuration:"
        dg_name = "dynamics generator"
        if (dyn_type == 'UNIT'):
            dg_name = "Hamiltonian"
        print "    Drift {}:".format(dg_name) 
        print dyn.drift_dyn_gen
        for j in range(dyn.num_ctrls):
            print "    Control {} {}:".format(j+1, dg_name)
            print dyn.Ctrl_dyn_gen[j]
        print "    Initial operator:"
        print dyn.initial
        print "    Target operator:"
        print dyn.target
    
    # Generate pulses for each ctrl
    init_amps = np.zeros([num_tslots, dyn.num_ctrls])
    for j in range(dyn.num_ctrls):
        init_amps[:, j] = p_gen.gen_pulse()

    # Initialise the starting amplitudes
    dyn.initialize_controls(init_amps)
    
    if (out_file_ext is not None):
        # Save initial amplitudes to a text file
        pulsefile = "ctrl_amps_initial_" + out_file_ext
        dyn.save_amps(pulsefile)
        if (msg_level >= 1):
            print "Initial amplitudes output to file: " + pulsefile
        
    # Start the optimisatiif (msg_level >= 1):on
    result = optim.run_optimization()

    if (out_file_ext is not None):
        # Save final amplitudes to a text file
        pulsefile = "ctrl_amps_final_" + out_file_ext
        dyn.save_amps(pulsefile)
        if (msg_level >= 1):
            print "Final amplitudes output to file: " + pulsefile
        
    return result

###############################################################################

def optimize_pulse_unitary(\
            H_d, H_c, U_0, U_targ, \
            num_tslots=None, evo_time=None, tau=None, \
            amp_lbound=-np.Inf, amp_ubound=np.Inf, \
            fid_err_targ=1e-10, min_grad=1e-10, \
            max_iter=500, max_wall_time=180, \
            optim_alg='LBFGSB', phase_option='PSU', amp_update_mode='ALL', \
            init_pulse_type='RND', \
            msg_level=0, out_file_ext='.txt', gen_stats=False):
                
    """
    Optimise a ctrl pulse to minimise the fidelity error, assuming that
    the dynamics of the system are generated by unitary operators.
    This function is simply a wrapper for optimize_pulse, where the 
    appropriate options for unitary dynamics are chosen and the parameter
    names are in the format familiar to unitary dynamics
    The dynamics of the system  in any given timeslot are governed 
    by the combined Hamiltonian, 
    i.e. the sum of the H_d + ctrl_amp[j]*H_c[j]
    The ctrl pulse is an [n_ts, len(Ctrls)] array of piecewise amplitudes
    Starting from an intital (typically random) pulse, 
    a multivariable optimisation algorithm attempts to determines the 
    optimal values for the ctrl pulse to minimise the fidelity error
    The maximum fidelity for a unitary system is 1, i.e. when the
    time evolution resulting from the pulse is equivalent to the target.
    And therefore the fidelity error is 1 - fidelity
    Parameters:
        H_d - Drift (aka system) the underlying Hamiltonian of the system
        H_c - a list of control Hamiltonians. These are scaled by
                the amplitudes to alter the overall dynamics
        U_0 - starting point for the evolution. 
                Typically the identity matrix
        U_targ - target transformation, e.g. gate or state, 
                for the time evolution
        num_tslots - number of timeslots
        evo_time - total time for the evolution
        tau - array of durations for the timeslots.
                if this is given then num_tslots and evo_time are dervived
                from it
        amp_lbound / amp_ubound - lower and upper boundaries for the ctrl
                ctrl amplitudes. Can be a scalar value applied to all ctrls
                or a list of bounds for each ctrl
        fid_err_targ - Fidelity error target. Pulse optimisation will 
                terminate when the fidelity error falls below this value
        mim_grad - Minimum gradient. When the sum of the squares of the
                gradients wrt to the ctrl amplitudes falls below this
                value, the optimisation terminates, assuming local minima
        max_iter - Maximum number of iterations of the optimisation algorithm
        max_wall_time - Maximum allowed elapsed time for the 
                optimisation algorithm
        optim_alg - Multi-variable optimisation algorithm
                options are BFGS, LBFGSB
                (see Optimizer classes for details)
        phase_option - determines how global phase is treated in fidelity
                calculations. Options:
                    PSU - global phase ignored
                    SU - global phase included
        amp_update_mode - determines whether propagators are calculated
                Options: ALL, DYNAMIC (needs work)
                (See TimeslotComputer classes for details)
        init_pulse_type - type / shape of pulse(s) used to initialise the
                the ctrl amplitudes. Options include:
                    RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
                (see PulseGen classes for details)
        msg_level - level of messaging output to the stdout.
                0 -> no messaging, up to about 5 for lots of debug messages
        out_file_ext - files containing the initial and final ctrl pulse
                amplitudes are saved to the current directory. 
                The default name will be postfixed with this extension
                Setting this to None will suppress the ouput of files
        gen_stats - if set to True then statistics for the optimisation
                run will be generated - accessible through attributes 
                of the stats object
    Returns:
        Returns an OptimResult object which has attributes giving the
        reason for termination, final fidelity error, final evolution,
        final amplitudes, statistics etc
    """
    return optimize_pulse(drift=H_d, Ctrls=H_c, initial=U_0, target=U_targ, \
            num_tslots=num_tslots, evo_time=evo_time, tau=tau, 
            amp_lbound=amp_lbound, amp_ubound=amp_ubound, \
            fid_err_targ=fid_err_targ, min_grad=min_grad, \
            max_iter=max_iter, max_wall_time=max_wall_time, \
            optim_alg=optim_alg, dyn_type='UNIT', phase_option=phase_option, \
            amp_update_mode=amp_update_mode, init_pulse_type=init_pulse_type, \
            msg_level=msg_level, out_file_ext=out_file_ext, \
            gen_stats=gen_stats)
            
            
###############################################################################
            
def create_pulse_optimizer(\
            drift, Ctrls, initial, target, \
            num_tslots=None, evo_time=None, tau=None, \
            amp_lbound=-np.Inf, amp_ubound=np.Inf, \
            fid_err_targ=1e-10, min_grad=1e-10, \
            max_iter=500, max_wall_time=180, \
            optim_alg='LBFGSB', dyn_type='GEN_MAT', prop_type='DEF', \
            fid_type='DEF', phase_option=None, amp_update_mode='ALL', \
            init_pulse_type='RND', \
            msg_level=0, gen_stats=False):
                
    """
    Generates the objects of the appropriate subclasses 
    required for the pulse optmisation based on the parameters given
    Note this method may be preferable to calling optimize_pulse
    if more detailed configuration is required before running the
    optmisation algorthim, or the algorithm will be run many times,
    for instances when trying to finding global the optimum or 
    minimum time optimisation
    
    Parameters:
        drift - (aka system) the underlying dynamics generator of the system
        Ctrls - a list of control dynamics generators. These are scalling by
                the amplitudes to alter the overall dynamics
        initial - starting point for the evolution. 
                Typically the identity matrix
        target - target transformation, e.g. gate or state, 
                for the time evolution
        num_tslots - number of timeslots
        evo_time - total time for the evolution
        tau - array of durations for the timeslots.
                if this is given then num_tslots and evo_time are dervived
                from it
        amp_lbound / amp_ubound - lower and upper boundaries for the ctrl
                ctrl amplitudes. Can be a scalar value applied to all ctrls
                or a list of bounds for each ctrl
        fid_err_targ - Fidelity error target. Pulse optimisation will 
                terminate when the fidelity error falls below this value
        mim_grad - Minimum gradient. When the sum of the squares of the
                gradients wrt to the ctrl amplitudes falls below this
                value, the optimisation terminates, assuming local minima
        max_iter - Maximum number of iterations of the optimisation algorithm
        max_wall_time - Maximum allowed elapsed time for the 
                optimisation algorithm
        optim_alg - Multi-variable optimisation algorithm
                options are BFGS, LBFGSB
                (see Optimizer classes for details)
        dyn_type - Dynamics type, i.e. the type of matrix used to describe
                the dynamics. Options are UNIT, GEN_MAT, SYMPL
                (see Dynamics classes for details)
        prop_type - i.e. the method used to calculate the
                propagtors and propagtor gradient for each timeslot
                options are APPROX, DIAG, FRECHET, AUG_MAT
                (see PropagatorComputer classes for details)
        fid_type - Fidelity error and (fidelity error gradient) computation
                method. Options are UNIT, TRACEDIFF, TD_APPROX
                (See FideliyComputer classes for details)
        phase_option - determines how global phase is treated in fidelity
                calculations (fid_type='UNIT' only). Options:
                    PSU - global phase ignored
                    SU - global phase included
        amp_update_mode - determines whether propagators are calculated
                Options: ALL, DYNAMIC (needs work)
                (See TimeslotComputer classes for details)
        init_pulse_type - type / shape of pulse(s) used to initialise the
                the ctrl amplitudes. Options include:
                    RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
                (see PulseGen classes for details)
        msg_level - level of messaging output to the stdout.
                0 -> no messaging, up to about 5 for lots of debug messages
        gen_stats - if set to True then a stats object will be created
                and added to Dynamics and Optimiser objects, which will mean
                that statistics will be generated for the optimisation run

    Returns:
        an Optimizer object, through which the 
        Config, Dynamics, PulseGen, and TerminationConditions objects
        can be accessed as attributes. 
        The PropagatorComputer, FidelityComputer and TimeslotComputer objects
        can be accessed as attributes of the Dynamics object, e.g.
            optimizer.dynamics.fid_computer
        The optimisation can be run through the optimizer.run_optim
    """

    this_func = 'create_pulse_optimizer'
    cfg = optimconfig.OptimConfig()
    cfg.msg_level = msg_level
    cfg.optim_alg = optim_alg
    cfg.amp_update_mode = amp_update_mode
    cfg.dyn_type = dyn_type
    cfg.prop_type = prop_type
    cfg.fid_type = fid_type
    cfg.pulse_type = init_pulse_type
    cfg.phase_option = phase_option
    
    # Number of time slots
    if (cfg.dyn_type == 'GEN_MAT' or cfg.dyn_type == None \
                or cfg.dyn_type == ''):
        dyn = dynamics.Dynamics_GenMat(cfg)
    elif (cfg.dyn_type == 'UNIT'):
        dyn = dynamics.Dynamics_Unitary(cfg)
    elif (cfg.dyn_type == 'SYMPL'):
        dyn = dynamics.Dynamics_Sympl(cfg)
    else:
        raise errors.UsageError(this_func, 'No option for cfg.dyn_type: ' + \
                                        cfg.dyn_type)
    
    if (cfg.prop_type == 'DEF' or cfg.prop_type == None \
                or cfg.prop_type == ''):
        # Do nothing use the default for the Dynamics
        pass
    elif (cfg.prop_type == 'APPROX'):
        if (not isinstance(dyn.prop_computer, propcomp.PropComp_ApproxGrad)):
            dyn.prop_computer = propcomp.PropComp_ApproxGrad(dyn)
    elif (cfg.prop_type == 'DIAG'):
        if (not isinstance(dyn.prop_computer, propcomp.PropComp_Diag)):
            dyn.prop_computer = propcomp.PropComp_Diag(dyn)
    elif (cfg.prop_type == 'AUG_MAT'):
        if (not isinstance(dyn.prop_computer, propcomp.PropComp_AugMat)):
            dyn.prop_computer = propcomp.PropComp_AugMat(dyn)
    elif (cfg.prop_type == 'FRECHET'):
        if (not isinstance(dyn.prop_computer, propcomp.PropComp_Frechet)):
            dyn.prop_computer = propcomp.PropComp_Frechet(dyn)
    else:
        raise errors.UsageError(this_func, 'No option for cfg.prop_type: ' + \
                                        cfg.prop_type)
                                        
    # The settig of the fidelity computer is unecessary, as the appropriate one
    # will be set as the default for the Dynamics, but this is here for testing
    if (cfg.fid_type == 'DEF' or cfg.fid_type == None or cfg.fid_type == ''):
        # None given, use the default for the Dynamics
        pass
    elif (cfg.fid_type == 'TRACEDIFF'):
        if (not isinstance(dyn.fid_computer, fidcomp.FidComp_TraceDiff)):
            dyn.fid_computer = fidcomp.FidComp_TraceDiff(dyn)
    elif (cfg.fid_type == 'TDAPPROX'):
        if (not isinstance(dyn.fid_computer, fidcomp.FidComp_TraceDiff_ApproxGrad)):
            dyn.fid_computer = fidcomp.FidComp_TraceDiff_ApproxGrad(dyn)
    elif (cfg.fid_type == 'UNIT'):
        if (not isinstance(dyn.fid_computer, fidcomp.FidComp_Unitary)):
            dyn.fid_computer = fidcomp.FidComp_Unitary(dyn)
    else:
        raise errors.UsageError(this_func, 'No option for cfg.fid_type: ' + \
                                        cfg.fid_type)
    
    if (isinstance(dyn.fid_computer, fidcomp.FidComp_Unitary)):
        dyn.fid_computer.set_phase_option(phase_option)
    
    # Create the Optimiser obect
    # The class of the object will determine which multivar optimisation
    # algorithm is used                                
    if (cfg.optim_alg=='BFGS'):
        optim = optimizer.Optimizer_BFGS(cfg, dyn)
    elif (cfg.optim_alg=='LBFGSB'):
        optim = optimizer.Optimizer_LBFGSB(cfg, dyn)
    else:
        raise errors.UsageError(this_func, 'No option for cfg.optim_alg: ' + \
                                        cfg.optim_alg)
    
    # Create the termination conditions object    
    tc = termcond.TerminationConditions()
    tc.fid_err_targ = fid_err_targ
    tc.min_gradient_norm = min_grad
    tc.max_iterations = max_iter
    tc.max_wall_time = max_wall_time
    optim.termination_conditions = tc
    
    if (gen_stats):
        # Create a stats object
        # Note that stats object is optional
        # if the Dynamics and Optimizer stats attribute is not set
        # then no stats will be collected, which could improve performance
        if (cfg.amp_update_mode == 'DYNAMIC'):
            sts = stats.Stats_DynTsUpdate()
        else:
            sts = stats.Stats()
            
        dyn.stats = sts
        optim.stats = sts
    
    # Configure the dynamics
    dyn.drift_dyn_gen = drift
    dyn.Ctrl_dyn_gen = Ctrls
    dyn.initial = initial
    dyn.target = target
    if (tau is None):
        # Check that parameters have been supplied to generate the 
        # timeslot durations
        try:
            dt = evo_time / num_tslots
        except:
            m = "Either the timeslot durations should be supplied as an" + \
                " array 'tau' or the number of timeslots num_tslots" + \
                " and the evolution time evo_time must be given."
            raise errors.UsageError(this_func, m)
            
        dyn.num_tslots = num_tslots
        dyn.evo_time = evo_time
    else:
        dyn.tau = tau
    # this function is called, so that the num_ctrls attribute will be set
    n_ctrls = dyn.get_num_ctrls()
    
    # Create a pulse generator of the type specified
    p_gen = pulsegen.create_pulse_gen(pulse_type=init_pulse_type, dyn=dyn)
    # If the pulse is a periodic type, then set the pulse to be one complete
    # wave
    if (isinstance(p_gen, pulsegen.PulseGen_Periodic)):
        p_gen.num_waves = 1.0
    optim.pulse_generator = p_gen
    
    if (msg_level >= 2):
        print "Optimisation config summary..."
        print "  object classes:"
        print "    optimizer: " + optim.__class__.__name__
        print "    dynamics: " + dyn.__class__.__name__
        print "    tslotcomp: " + dyn.tslot_computer.__class__.__name__
        print "    fidcomp: " + dyn.fid_computer.__class__.__name__
        print "    propcomp: " + dyn.prop_computer.__class__.__name__
        print "    pulsegen: " + p_gen.__class__.__name__
    
    return optim
    
    
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

"""
Wrapper functions that will manage the creation of the objects,
build the configuration, and execute the algorithm required to optimise
a set of ctrl pulses for a given (quantum) system.
The fidelity error is some measure of distance of the system evolution
from the given target evolution in the time allowed for the evolution.
The functions minimise this fidelity error wrt the piecewise control
amplitudes in the timeslots
"""
import numpy as np
# QuTiP
from qutip import Qobj
import qutip.logging as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
import qutip.control.errors as errors
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.control.pulsegen as pulsegen
#import qutip.control.pulsegencrab as pulsegencrab

def _upper_safe(s):
    try:
        s = s.upper()
    except:
        pass
    return s

def optimize_pulse(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=-np.Inf, amp_ubound=np.Inf,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        optim_alg='LBFGSB', max_metric_corr=10, accuracy_factor=1e7,
        dyn_type='GEN_MAT', prop_type='DEF',
        fid_type='DEF', phase_option=None, fid_err_scale_factor=None,
        amp_update_mode='ALL',
        init_pulse_type='RND', pulse_scaling=1.0, pulse_offset=0.0,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False):
    """
    Optimise a control pulse to minimise the fidelity error.
    The dynamics of the system in any given timeslot are governed
    by the combined dynamics generator,
    i.e. the sum of the drift+ctrl_amp[j]*ctrls[j]
    The control pulse is an [n_ts, len(ctrls)] array of piecewise amplitudes
    Starting from an intital (typically random) pulse,
    a multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The fidelity error is some measure of distance of the system evolution
    from the given target evolution in the time allowed for the evolution.

    Parameters
    ----------

    drift : Qobj
        the underlying dynamics generator of the system

    ctrls : List of Qobj
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    optim_alg : string
        Multi-variable optimisation algorithm
        options are BFGS, LBFGSB
        (see Optimizer classes for details)

    max_metric_corr : integer
        The maximum number of variable metric corrections used to define
        the limited memory matrix. That is the number of previous
        gradient values that are used to approximate the Hessian
        see the scipy.optimize.fmin_l_bfgs_b documentation for description
        of m argument
        (used only in L-BFGS-B)

    accuracy_factor : float
        Determines the accuracy of the result.
        Typical values for accuracy_factor are: 1e12 for low accuracy;
        1e7 for moderate accuracy; 10.0 for extremely high accuracy
        scipy.optimize.fmin_l_bfgs_b factr argument.
        (used only in L-BFGS-B)

    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:
            PSU - global phase ignored
            SU - global phase included

    fid_err_scale_factor : float
        (used in TRACEDIFF FidelityComputer and subclasses only)
        The fidelity error calculated is of some arbitary scale. This
        factor can be used to scale the fidelity error such that it may
        represent some physical measure
        If None is given then it is caculated as 1/2N, where N
        is the dimension of the drift.

    amp_update_mode : string
        determines whether propagators are calculated
        Options: DEF, ALL, DYNAMIC (needs work)
        DEF will use the default for the specific dyn_type
        (See TimeslotComputer classes for details)

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes. Options include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
        (see PulseGen classes for details)

    pulse_scaling : float
        Linear scale factor for generated pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any initial pulses generated.

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    """

    # The parameters are checked in create_pulse_optimizer
    # so no need to do so here

    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    optim = create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ, min_grad=min_grad,
        max_iter=max_iter, max_wall_time=max_wall_time,
        optim_alg=optim_alg, max_metric_corr=max_metric_corr,
        accuracy_factor=accuracy_factor,
        dyn_type=dyn_type, prop_type=prop_type,
        fid_type=fid_type, phase_option=phase_option,
        fid_err_scale_factor=fid_err_scale_factor,
        amp_update_mode=amp_update_mode, init_pulse_type=init_pulse_type,
        pulse_scaling=pulse_scaling, pulse_offset=pulse_offset,
        log_level=log_level, gen_stats=gen_stats)

    dyn = optim.dynamics
    pgen = optim.pulse_generator

    if log_level <= logging.INFO:
        msg = "System configuration:\n"
        dg_name = "dynamics generator"
        if dyn_type == 'UNIT':
            dg_name = "Hamiltonian"
        msg += "Drift {}:\n".format(dg_name)
        msg += str(dyn.drift_dyn_gen)
        for j in range(dyn.num_ctrls):
            msg += "\nControl {} {}:\n".format(j+1, dg_name)
            msg += str(dyn.ctrl_dyn_gen[j])
        msg += "\nInitial operator:\n"
        msg += str(dyn.initial)
        msg += "\nTarget operator:\n"
        msg += str(dyn.target)
        logger.info(msg)

    dyn.init_timeslots()
    # Generate initial pulses for each control
    init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])
    for j in range(dyn.num_ctrls):
        init_amps[:, j] = pgen.gen_pulse()

    # Initialise the starting amplitudes
    dyn.initialize_controls(init_amps)

    if out_file_ext is not None:
        # Save initial amplitudes to a text file
        pulsefile = "ctrl_amps_initial_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Initial amplitudes output to file: " + pulsefile)

    # Start the optimisation
    result = optim.run_optimization()

    if out_file_ext is not None:
        # Save final amplitudes to a text file
        pulsefile = "ctrl_amps_final_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Final amplitudes output to file: " + pulsefile)

    return result


def optimize_pulse_unitary(
        H_d, H_c, U_0, U_targ,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=-np.Inf, amp_ubound=np.Inf,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        optim_alg='LBFGSB', max_metric_corr=10, accuracy_factor=1e7,
        phase_option='PSU',
        amp_update_mode='ALL',
        init_pulse_type='RND', pulse_scaling=1.0, pulse_offset=0.0,
        log_level=logging.NOTSET, out_file_ext='.txt', gen_stats=False):

    """
    Optimise a control pulse to minimise the fidelity error, assuming that
    the dynamics of the system are generated by unitary operators.
    This function is simply a wrapper for optimize_pulse, where the
    appropriate options for unitary dynamics are chosen and the parameter
    names are in the format familiar to unitary dynamics
    The dynamics of the system  in any given timeslot are governed
    by the combined Hamiltonian,
    i.e. the sum of the H_d + ctrl_amp[j]*H_c[j]
    The control pulse is an [n_ts, len(ctrls)] array of piecewise amplitudes
    Starting from an intital (typically random) pulse,
    a multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The maximum fidelity for a unitary system is 1, i.e. when the
    time evolution resulting from the pulse is equivalent to the target.
    And therefore the fidelity error is 1 - fidelity

    Parameters
    ----------

    H_d : Qobj
        Drift (aka system) the underlying Hamiltonian of the system

    H_c : Qobj
        a list of control Hamiltonians. These are scaled by
        the amplitudes to alter the overall dynamics

    U_0 : Qobj
        starting point for the evolution.
        Typically the identity matrix

    U_targ : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    optim_alg : string
        Multi-variable optimisation algorithm
        options are BFGS, LBFGSB
        (see Optimizer classes for details)

    max_metric_corr : integer
        The maximum number of variable metric corrections used to define
        the limited memory matrix. That is the number of previous
        gradient values that are used to approximate the Hessian
        see the scipy.optimize.fmin_l_bfgs_b documentation for description
        of m argument
        (used only in L-BFGS-B)

    accuracy_factor : float
        Determines the accuracy of the result.
        Typical values for accuracy_factor are: 1e12 for low accuracy;
        1e7 for moderate accuracy; 10.0 for extremely high accuracy
        scipy.optimize.fmin_l_bfgs_b factr argument.
        (used only in L-BFGS-B)

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:
            PSU - global phase ignored
            SU - global phase included

    amp_update_mode : string
        determines whether propagators are calculated
        Options: DEF, ALL, DYNAMIC (needs work)
        DEF will use the default for the specific dyn_type
        (See TimeslotComputer classes for details)

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes. Options include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
        (see PulseGen classes for details)

    pulse_scaling : float
        Linear scale factor for generated pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any initial pulses generated.

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    """

    # check parameters here, as names are different than in
    # create_pulse_optimizer, so TypeErrors would be confusing

    if not isinstance(H_d, Qobj):
        raise TypeError("H_d must be a Qobj")

    if not isinstance(H_c, (list, tuple)):
        raise TypeError("H_c should be a list of Qobj")
    else:
        for ctrl in H_c:
            if not isinstance(ctrl, Qobj):
                raise TypeError("H_c should be a list of Qobj")

    if not isinstance(U_0, Qobj):
        raise TypeError("U_0 must be a Qobj")

    if not isinstance(U_targ, Qobj):
        raise TypeError("U_targ must be a Qobj")

    return optimize_pulse(drift=H_d, ctrls=H_c, initial=U_0, target=U_targ,
                          num_tslots=num_tslots, evo_time=evo_time, tau=tau,
                          amp_lbound=amp_lbound, amp_ubound=amp_ubound,
                          fid_err_targ=fid_err_targ, min_grad=min_grad,
                          max_iter=max_iter, max_wall_time=max_wall_time,
                          optim_alg=optim_alg, max_metric_corr=max_metric_corr,
                          accuracy_factor=accuracy_factor,
                          dyn_type='UNIT', phase_option=phase_option,
                          amp_update_mode=amp_update_mode,
                          init_pulse_type=init_pulse_type,
                          pulse_scaling=pulse_scaling,
                          pulse_offset=pulse_offset,
                          log_level=log_level, out_file_ext=out_file_ext,
                          gen_stats=gen_stats)


def create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=-np.Inf, amp_ubound=np.Inf,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        alg='GRAPE', alg_options=None,
        optim_method='DEF', method_options=None,
        optim_alg=None, max_metric_corr=10, accuracy_factor=1e7,
        dyn_type='GEN_MAT', prop_type='DEF',
        fid_type='DEF', phase_option=None, fid_err_scale_factor=None,
        amp_update_mode='ALL',
        init_pulse_type='DEF', pulse_scaling=1.0, pulse_offset=0.0,
        ramping_pulse_type=None, ramping_pulse_options=None,
        log_level=logging.NOTSET, gen_stats=False):

    """
    Generate the objects of the appropriate subclasses
    required for the pulse optmisation based on the parameters given
    Note this method may be preferable to calling optimize_pulse
    if more detailed configuration is required before running the
    optmisation algorthim, or the algorithm will be run many times,
    for instances when trying to finding global the optimum or
    minimum time optimisation

    Parameters
    ----------

    drift : Qobj
        the underlying dynamics generator of the system

    ctrls : List of Qobj
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the optimisation algorithm
        
    alg : string
        Algorithm to use in pulse optimisation.
        Options are:
            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    alg_options : Dictionary
        options that are specific to the algorithm see above
        
    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error
        Note that FMIN, FMIN_BFGS & FMIN_L_BFGS_B will all result
        in calling these specific scipy.optimize methods
        Note the LBFGSB is equivalent to FMIN_L_BFGS_B for backwards 
        capatibility reasons.
        Supplying DEF will given alg dependent result:
            GRAPE - Default optim_method is FMIN_L_BFGS_B
            CRAB - Default optim_method is Nelder-Mead
        
    method_options : Dictionary
        Options for the optim_method. 
        Note that where there is an equivalent attribute of this instance
        or the termination_conditions (for example maxiter)
        it will override an value in these options
        
    optim_alg : string
        Multi-variable optimisation algorithm
        options are BFGS, LBFGSB
        (see Optimizer classes for details)

    max_metric_corr : integer
        The maximum number of variable metric corrections used to define
        the limited memory matrix. That is the number of previous
        gradient values that are used to approximate the Hessian
        see the scipy.optimize.fmin_l_bfgs_b documentation for description
        of m argument
        (used only in L-BFGS-B)

    accuracy_factor : float
        Determines the accuracy of the result.
        Typical values for accuracy_factor are: 1e12 for low accuracy;
        1e7 for moderate accuracy; 10.0 for extremely high accuracy
        scipy.optimize.fmin_l_bfgs_b factr argument.
        (used only in L-BFGS-B)

    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:
            PSU - global phase ignored
            SU - global phase included

    fid_err_scale_factor : float
        (used in TRACEDIFF FidelityComputer and subclasses only)
        The fidelity error calculated is of some arbitary scale. This
        factor can be used to scale the fidelity error such that it may
        represent some physical measure
        If None is given then it is caculated as 1/2N, where N
        is the dimension of the drift.

    amp_update_mode : string
        determines whether propagators are calculated
        Options: DEF, ALL, DYNAMIC (needs work)
        DEF will use the default for the specific dyn_type
        (See TimeslotComputer classes for details)

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes. 
        Options (GRAPE) include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
        DEF is RND
        (see PulseGen classes for details)
        for the CRAB the this determines the pulse generator type for each
        control. Currently only option is: CRAB_FOURIER

    pulse_scaling : float
        Linear scale factor for generated pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any initial pulses generated.

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN
        Note value should be set using set_log_level

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
        Instance of an Optimizer, through which the
        Config, Dynamics, PulseGen, and TerminationConditions objects
        can be accessed as attributes.
        The PropagatorComputer, FidelityComputer and TimeslotComputer objects
        can be accessed as attributes of the Dynamics object, e.g.
            optimizer.dynamics.fid_computer
        The optimisation can be run through the optimizer.run_optimization
    """

    # check parameters
    if not isinstance(drift, Qobj):
        raise TypeError("drift must be a Qobj")
    else:
        drift = drift.full()

    if not isinstance(ctrls, (list, tuple)):
        raise TypeError("ctrls should be a list of Qobj")
    else:
        j = 0
        for ctrl in ctrls:
            if not isinstance(ctrl, Qobj):
                raise TypeError("ctrls should be a list of Qobj")
            else:
                ctrls[j] = ctrl.full()
                j += 1

    if not isinstance(initial, Qobj):
        raise TypeError("initial must be a Qobj")
    else:
        initial = initial.full()

    if not isinstance(target, Qobj):
        raise TypeError("target must be a Qobj")
    else:
        target = target.full()
        
    # Deprecated parameter management
    if optim_alg:
        optim_method = optim_alg
      
    # set algorithm defaults
    alg_up = _upper_safe(alg)
    if alg is None:
        raise errors.UsageError(
            "Optimisation algorithm must be specified through 'alg' parameter")
    elif alg_up == 'GRAPE':
        if optim_method is None or optim_method.upper() == 'DEF':
            optim_method = 'FMIN_L_BFGS_B'
        if init_pulse_type is None or init_pulse_type.upper() == 'DEF':
            init_pulse_type = 'RND'
    elif alg_up == 'CRAB':
        if optim_method is None or optim_method.upper() == 'DEF':
            optim_method = 'Nelder-Mead'
        if prop_type is None or prop_type.upper() == 'DEF':
            prop_type = 'APPROX'
        #Use equivalent parameters for alg_options
        if init_pulse_type is None or init_pulse_type.upper() == 'DEF':
            # Ignore as there are no other equivalent params
            pass
        else:
            if not isinstance(alg_options, dict):
                alg_options = {}
            if not 'guess_pulse_type' in alg_options:
                alg_options['guess_pulse_type'] = init_pulse_type
            if not 'guess_pulse_scaling' in alg_options:
                alg_options['guess_pulse_scaling'] = pulse_scaling
            if not 'guess_pulse_offset' in alg_options:
                alg_options['guess_pulse_offset'] = pulse_offset
                
    else:
        raise errors.UsageError(
            "No option for pulse optimisation algorithm alg={}".format(alg))

    cfg = optimconfig.OptimConfig()
    cfg.optim_alg = optim_alg
    cfg.max_metric_corr = max_metric_corr
    cfg.accuracy_factor = accuracy_factor
    cfg.amp_update_mode = amp_update_mode
    cfg.dyn_type = dyn_type
    cfg.prop_type = prop_type
    cfg.fid_type = fid_type
    cfg.pulse_type = init_pulse_type
    cfg.phase_option = phase_option
    cfg.amp_lbound = amp_lbound
    cfg.amp_ubound = amp_ubound

    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    cfg.log_level = log_level

    # Create the Dynamics instance
    if dyn_type == 'GEN_MAT' or dyn_type is None or dyn_type == '':
        dyn = dynamics.DynamicsGenMat(cfg)
    elif dyn_type == 'UNIT':
        dyn = dynamics.DynamicsUnitary(cfg)
    elif dyn_type == 'SYMPL':
        dyn = dynamics.DynamicsSymplectic(cfg)
    else:
        raise errors.UsageError("No option for dyn_type: " + dyn_type)

    # Create the PropagatorComputer instance
    # The default will be typically be the best option
    if prop_type == 'DEF' or prop_type is None or prop_type == '':
        # Do nothing use the default for the Dynamics
        pass
    elif prop_type == 'APPROX':
        if not isinstance(dyn.prop_computer, propcomp.PropCompApproxGrad):
            dyn.prop_computer = propcomp.PropCompApproxGrad(dyn)
    elif prop_type == 'DIAG':
        if not isinstance(dyn.prop_computer, propcomp.PropCompDiag):
            dyn.prop_computer = propcomp.PropCompDiag(dyn)
    elif prop_type == 'AUG_MAT':
        if not isinstance(dyn.prop_computer, propcomp.PropCompAugMat):
            dyn.prop_computer = propcomp.PropCompAugMat(dyn)
    elif prop_type == 'FRECHET':
        if not isinstance(dyn.prop_computer, propcomp.PropCompFrechet):
            dyn.prop_computer = propcomp.PropCompFrechet(dyn)
    else:
        raise errors.UsageError("No option for prop_type: " + prop_type)

    # Create the FidelityComputer instance
    # The default will be typically be the best option
    # Note: the FidCompTraceDiffApprox is a subclass of FidCompTraceDiff
    # so need to check this type first
    fid_type_up = _upper_safe(fid_type)
    if fid_type_up == 'DEF' or fid_type_up is None or fid_type_up == '':
        # None given, use the default for the Dynamics
        pass
    elif fid_type_up == 'TDAPPROX':
        if not isinstance(dyn.fid_computer, fidcomp.FidCompTraceDiffApprox):
            dyn.fid_computer = fidcomp.FidCompTraceDiffApprox(dyn)
    elif fid_type_up == 'TRACEDIFF':
        if not isinstance(dyn.fid_computer, fidcomp.FidCompTraceDiff):
            dyn.fid_computer = fidcomp.FidCompTraceDiff(dyn)
    elif fid_type_up == 'UNIT':
        if not isinstance(dyn.fid_computer, fidcomp.FidCompUnitary):
            dyn.fid_computer = fidcomp.FidCompUnitary(dyn)
    else:
        raise errors.UsageError("No option for fid_type: " + fid_type)

    if isinstance(dyn.fid_computer, fidcomp.FidCompUnitary):
        # If phase option is None take the default of PSU
        if not phase_option:
            phase_option = 'PSU'
        dyn.fid_computer.set_phase_option(phase_option)

    if isinstance(dyn.fid_computer, fidcomp.FidCompTraceDiff):
        dyn.fid_computer.scale_factor = fid_err_scale_factor

    # Create the Optimiser instance
    optim_method_up = _upper_safe(optim_method)
    if optim_method is None or optim_method_up == '':
        raise errors.UsageError("Optimisation method must be specified "
                                "via 'optim_method' parameter")
    elif optim_method_up == 'FMIN_BFGS':
        optim = optimizer.OptimizerBFGS(cfg, dyn)
    elif optim_method_up == 'LBFGSB' or optim_method_up == 'FMIN_L_BFGS_B':
        optim = optimizer.OptimizerLBFGSB(cfg, dyn)
    else:
        # Assume that the optim_method is a valid
        #scipy.optimize.minimize method
        # Choose an optimiser based on the algorithm
        if alg_up == 'CRAB':
            optim = optimizer.OptimizerCrab(cfg, dyn)
        else:
            optim = optimizer.Optimizer(cfg, dyn)
            
    optim.method = optim_method
    optim.method_options = method_options

    # Create the TerminationConditions instance
    tc = termcond.TerminationConditions()
    tc.fid_err_targ = fid_err_targ
    tc.min_gradient_norm = min_grad
    tc.max_iterations = max_iter
    tc.max_wall_time = max_wall_time
    optim.termination_conditions = tc

    if gen_stats:
        # Create a stats object
        # Note that stats object is optional
        # if the Dynamics and Optimizer stats attribute is not set
        # then no stats will be collected, which could improve performance
        if amp_update_mode == 'DYNAMIC':
            sts = stats.StatsDynTsUpdate()
        else:
            sts = stats.Stats()

        dyn.stats = sts
        optim.stats = sts

    # Configure the dynamics
    dyn.drift_dyn_gen = drift
    dyn.ctrl_dyn_gen = ctrls
    dyn.initial = initial
    dyn.target = target
    if tau is None:
        # Check that parameters have been supplied to generate the
        # timeslot durations
        try:
            evo_time / num_tslots
        except:
            raise errors.UsageError(
                "Either the timeslot durations should be supplied as an "
                "array 'tau' or the number of timeslots 'num_tslots' "
                "and the evolution time 'evo_time' must be given.")

        dyn.num_tslots = num_tslots
        dyn.evo_time = evo_time
    else:
        dyn.tau = tau

    # this function is called, so that the num_ctrls attribute will be set
    n_ctrls = dyn.get_num_ctrls()

    ramping_pgen = None
    if ramping_pulse_type:
        ramping_pgen = pulsegen.create_pulse_gen(
                            pulse_type=ramping_pulse_type, dyn=dyn, 
                            pulse_options=ramping_pulse_options)
    if alg_up == 'CRAB':
        # Create a pulse generator for each ctrl      
        num_coeffs = alg_options.get('num_coeffs')
        init_coeff_scaling = alg_options.get('init_coeff_scaling')
        guess_pulse_type = alg_options.get('guess_pulse_type')
        if guess_pulse_type:
            guess_pgen = pulsegen.create_pulse_gen(
                                pulse_type=guess_pulse_type, dyn=dyn)
            if 'guess_pulse_scaling' in alg_options:
                guess_pgen.scaling = alg_options['guess_pulse_scaling']
            if 'guess_pulse_offset' in alg_options:
                guess_pgen.offset = alg_options['guess_pulse_offset']
            guess_pulse_action = alg_options.get('guess_pulse_action')
            
        optim.pulse_generator = []
        for j in range(n_ctrls):
            crab_pgen = pulsegen.PulseGenCrabFourier(
                                dyn=dyn, num_coeffs=num_coeffs)
            if init_coeff_scaling is not None:
                crab_pgen.scaling = init_coeff_scaling
                
            if guess_pulse_type:
                crab_pgen.guess_pulse = guess_pgen.gen_pulse()
                if guess_pulse_action:
                    crab_pgen.guess_pulse_action = guess_pulse_action
                
            if ramping_pgen:
                crab_pgen.ramping_pulse = ramping_pgen.gen_pulse()

            optim.pulse_generator.append(crab_pgen)
        #This is just for the debug message now
        pgen = optim.pulse_generator[0]
            
    else:
        # Create a pulse generator of the type specified
        pgen = pulsegen.create_pulse_gen(pulse_type=init_pulse_type, dyn=dyn)
        pgen.scaling = pulse_scaling
        pgen.offset = pulse_offset
        pgen.lbound = amp_lbound
        pgen.ubound = amp_ubound
        if ramping_pgen:
            crab_pgen.ramping_pulse = ramping_pgen.gen_pulse()

        # If the pulse is a periodic type, then set the pulse to be one complete
        # wave
        if isinstance(pgen, pulsegen.PulseGenPeriodic):
            pgen.num_waves = 1.0
        optim.pulse_generator = pgen

    if log_level <= logging.DEBUG:
        logger.debug(
            "Optimisation config summary...\n"
            "  object classes:\n"
            "    optimizer: " + optim.__class__.__name__ +
            "\n    dynamics: " + dyn.__class__.__name__ +
            "\n    tslotcomp: " + dyn.tslot_computer.__class__.__name__ +
            "\n    fidcomp: " + dyn.fid_computer.__class__.__name__ +
            "\n    propcomp: " + dyn.prop_computer.__class__.__name__ +
            "\n    pulsegen: " + pgen.__class__.__name__)

    return optim

def opt_pulse_crab(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=-np.Inf, amp_ubound=np.Inf,
        fid_err_targ=1e-10,
        max_iter=500, max_wall_time=180,
        num_coeffs=None, init_coeff_scaling=1.0, 
        optim_method='Nelder-Mead', method_options=None,
        dyn_type='GEN_MAT', prop_type='DEF',
        fid_type='DEF', phase_option=None, fid_err_scale_factor=None,
        guess_pulse_type=None, guess_pulse_scaling=1.0, guess_pulse_offset=0.0,
        guess_pulse_action='modulate',
        ramping_pulse_type=None, ramping_pulse_options=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False):
    """
    Optimise a control pulse to minimise the fidelity error.
    The dynamics of the system in any given timeslot are governed
    by the combined dynamics generator,
    i.e. the sum of the drift+ctrl_amp[j]*ctrls[j]
    The control pulse is an [n_ts, len(ctrls)] array of piecewise amplitudes.
    The CRAB algorithm uses basis function coefficents as the variables to
    optimise. It does NOT use any gradient function.
    A multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The fidelity error is some measure of distance of the system evolution
    from the given target evolution in the time allowed for the evolution.

    Parameters
    ----------

    drift : Qobj
        the underlying dynamics generator of the system

    ctrls : List of Qobj
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    optim_method : string
        Multi-variable optimisation method
        The only tested option is 'Nelder-mead'
        In theory any non-gradient method implemented in 
        scipy.optimize.mininize could be used.

    method_options : Dictionary
        Options for the optim_method. 
        Note that where there is an equivalent attribute of this instance
        or the termination_conditions (for example maxiter)
        it will override an value in these options
        
    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)

    fid_type : string
        Fidelity error computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:
            PSU - global phase ignored
            SU - global phase included

    fid_err_scale_factor : float
        (used in TRACEDIFF FidelityComputer and subclasses only)
        The fidelity error calculated is of some arbitary scale. This
        factor can be used to scale the fidelity error such that it may
        represent some physical measure
        If None is given then it is caculated as 1/2N, where N
        is the dimension of the drift.

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes. Options include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
        (see PulseGen classes for details)

    coeff_scaling : float
        Linear scale factor for the basis coefficients
        By default initial pulses are generated with coefficients in the
        range (-1.0, 1.0). These will be scaled by this parameter

    coeff_offset : float
        Linear offset for the basis coefficients. 
        This value will be added to the coeffs when generating initial pulses.

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    """

    # The parameters are checked in create_pulse_optimizer
    # so no need to do so here

    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    # build the algorithm options       
    alg_options = {'num_coeffs':num_coeffs, 
                   'init_coeff_scaling':init_coeff_scaling,
                   'guess_pulse_type':guess_pulse_type,
                   'guess_pulse_scaling':guess_pulse_scaling,
                   'guess_pulse_offset':guess_pulse_offset,
                   'guess_pulse_action':guess_pulse_action}
#                   ,
#                   'pulse_ramping_type':ramping_type,
#                   'pulse_ramping_options':ramping_options}
                   
    optim = create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ,
        max_iter=max_iter, max_wall_time=max_wall_time,
        alg='CRAB', alg_options=alg_options,
        optim_method=optim_method, method_options=method_options,
        dyn_type=dyn_type, prop_type=prop_type,
        fid_type=fid_type, phase_option=phase_option,
        fid_err_scale_factor=fid_err_scale_factor,
        ramping_pulse_type=ramping_pulse_type, 
        ramping_pulse_options=ramping_pulse_options,
        log_level=log_level, gen_stats=gen_stats)

    dyn = optim.dynamics
    pgen = optim.pulse_generator

    if log_level <= logging.INFO:
        msg = "System configuration:\n"
        dg_name = "dynamics generator"
        if dyn_type == 'UNIT':
            dg_name = "Hamiltonian"
        msg += "Drift {}:\n".format(dg_name)
        msg += str(dyn.drift_dyn_gen)
        for j in range(dyn.num_ctrls):
            msg += "\nControl {} {}:\n".format(j+1, dg_name)
            msg += str(dyn.ctrl_dyn_gen[j])
        msg += "\nInitial operator:\n"
        msg += str(dyn.initial)
        msg += "\nTarget operator:\n"
        msg += str(dyn.target)
        logger.info(msg)

    dyn.init_timeslots()
    # Initialise the pulse generators and 
    # Generate initial pulses for each control
    init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])
    for j in range(dyn.num_ctrls):
        pgen = optim.pulse_generator[j]
        pgen.init_pulse()
        init_amps[:, j] = pgen.gen_pulse()

    # Initialise the starting amplitudes
    dyn.initialize_controls(init_amps)

    if out_file_ext is not None:
        # Save initial amplitudes to a text file
        pulsefile = "ctrl_amps_initial_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Initial amplitudes output to file: " + pulsefile)

    # Start the optimisation
    result = optim.run_optimization()

    if out_file_ext is not None:
        # Save final amplitudes to a text file
        pulsefile = "ctrl_amps_final_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Final amplitudes output to file: " + pulsefile)

    return result

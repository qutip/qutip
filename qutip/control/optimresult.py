# -*- coding: utf-8 -*-
# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Class containing the results of the pulse optimisation
"""

import numpy as np


class OptimResult(object):
    """
    Attributes give the result of the pulse optimisation attempt

    Attributes
    ----------
    termination_reason : string
        Description of the reason for terminating the optimisation

    fidelity : float
        final (normalised) fidelity that was achieved

    initial_fid_err : float
        fidelity error before optimisation starting
        
    fid_err : float
        final fidelity error that was achieved

    goal_achieved : boolean
        True is the fidely error achieved was below the target

    grad_norm_final : float
        Final value of the sum of the squares of the (normalised) fidelity
        error gradients

    grad_norm_min_reached : float
        True if the optimisation terminated due to the minimum value
        of the gradient being reached

    num_iter : integer
        Number of iterations of the optimisation algorithm completed

    max_iter_exceeded : boolean
        True if the iteration limit was reached
        
    max_fid_func_exceeded : boolean
        True if the fidelity function call limit was reached

    wall_time : float
        time elapsed during the optimisation

    wall_time_limit_exceeded  : boolean
        True if the wall time limit was reached

    time : array[num_tslots+1] of float
        Time are the start of each timeslot
        with the final value being the total evolution time

    initial_amps : array[num_tslots, n_ctrls]
        The amplitudes at the start of the optimisation

    final_amps : array[num_tslots, n_ctrls]
        The amplitudes at the end of the optimisation

    evo_full_final : Qobj
        The evolution operator from t=0 to t=T based on the final amps

    evo_full_initial : Qobj
        The evolution operator from t=0 to t=T based on the initial amps
        
    stats : Stats
        Object contaning the stats for the run (if any collected)
        
    optimizer : Optimizer
        Instance of the Optimizer used to generate the result
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.fidelity = 0.0
        self.initial_fid_err = np.Inf
        self.fid_err = np.Inf
        self.goal_achieved = False
        self.grad_norm_final = 0.0
        self.grad_norm_min_reached = False
        self.num_iter = 0
        self.max_iter_exceeded = False
        self.num_fid_func_calls = 0
        self.max_fid_func_exceeded = False
        self.wall_time = 0.0
        self.wall_time_limit_exceeded = False
        self.termination_reason = "not started yet"
        self.time = None
        self.initial_amps = None
        self.final_amps = None
        self.evo_full_final = None
        self.evo_full_initial = None
        self.stats = None
        self.optimizer = None

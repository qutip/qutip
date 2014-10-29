# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 18:49:51 2014
@author: Alexander Pitchford
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Class containing the results of the pulse optimisation
"""

import numpy as np

class OptimResult:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.goal = 1.0
        self.fidelity = 0.0
        self.fid_err = np.Inf
        self.goal_achieved = False
        self.grad_norm_final = 0.0
        self.grad_norm_min_reached = False
        self.grad_static = False
        self.num_iter = 0
        self.max_iter_exceeded = False
        self.wall_time = 0.0
        self.wall_time_limit_exceeded = False
        self.termination_reason = "not started yet"
        # Time are the start of each timeslot as array[num_tslots+1]
        # with the final value being the total evolaution time
        self.time = None
        # The amplitudes at the start of the optimisation
        self.initial_amps = None
        # The amplitudes at the end of the optimisation
        self.final_amps = None
        # The evolution operator from t=0 to t=T based on the final amps
        self.evo_full_final = None
        # Object contaning the stats for the run (if any collected)
        self.stats = None
        
    
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 17:31:04 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Classes containing termination conditions for the control pulse optimisation
i.e. attributes that will be checked during the optimisation, that
will determine if the algorithm has completed its task / exceeded limits
"""

class TerminationConditions:
    """
    Base class for all termination conditions
    Used to determine when to stop the optimisation algorithm
    Note different subclasses should be used to match the type of
    optimisation being used
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        # Target fidelity error
        self.fid_err_targ = None
        # goal fidelity, e.g. 1 - self.fid_err_targ
        # It's typical to set this for unitary systems
        self.fid_goal = None
        # maximum time for optimisation (seconds)
        self.max_wall_time = 60*60.0
        # Minimum normalised gradient for successful result
        # scipy.optimize.fmin_bfgs gtol argument.
        # Also checked explicitly in code
        self.min_gradient_norm = 1e-5
        # Max algorithm iterations
        self.max_iterations = 1e10
        # Max fidelity faunction calls
        # (LBFGSB only)
        self.max_fid_func_calls = 1e10
        

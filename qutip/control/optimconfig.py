# -*- coding: utf-8 -*-class OptimConfig:
"""
Created on Tue Feb 11 11:48:20 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Configuration parameters for qtrl optimisation
"""

import numpy as np

class OptimConfig:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # Level of messaging. Higher number means more messages
        self.msg_level = 0
        # Level of test ouptut files generated
        # NOTE: If >0 then sub directory 'test_out' must exist
        self.test_out_files = 0
        self.optim_alg = 'LBFGSB'
        self.dyn_type = ''
        self.fid_type = ''
        self.phase_option = 'PSU'
        self.amp_update_mode = 'ALL' #Alts: 'DYNAMIC'
        self.pulse_type = 'RND'
        ######################
        # Note the following parameteres are for constrained optimisation 
        # methods e.g. L-BFGS-B
        # These are the lower and upper bounds for the control amplitudes
        # note that one of these can be None for L-BFGS-B, but not both
        # these can be given as a list (one for each control) or a scalar
        # meaning the same value will be used for each control
        self.amp_lbound = -np.Inf
        self.amp_ubound = np.Inf
        # The maximum number of variable metric corrections used to define
        # the limited memory matrix
        # see the scipy.optimize.fmin_l_bfgs_b documentation for description
        # of m argument 
        self.max_metric_corr = 10
        # Determines the accuracy of the result.
        # Typical values for accuracy_factor are: 1e12 for low accuracy; 
        # 1e7 for moderate accuracy; 10.0 for extremely high accuracy
        # scipy.optimize.fmin_l_bfgs_b factr argument.
        self.optim_alg_acc_fact = 1e7
        # ####################
        
# create global instance
optimconfig = OptimConfig()




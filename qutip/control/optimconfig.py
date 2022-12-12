# -*- coding: utf-8 -*-
# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Configuration parameters for control pulse optimisation
"""

import numpy as np
# QuTiP logging
import qutip.logging_utils
logger = qutip.logging_utils.get_logger('qutip.control.optimconfig')
import qutip.control.io as qtrlio

class OptimConfig(object):
    """
    Configuration parameters for control pulse optimisation

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

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

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.log_level = logger.getEffectiveLevel()
        self.alg = 'GRAPE' # Alts: 'CRAB'
        # *** AJGP 2015-04-21: This has been replaced optim_method
        #self.optim_alg = 'LBFGSB'
        self.optim_method = 'DEF'
        self.dyn_type = 'DEF'
        self.fid_type = 'DEF'
        # *** AJGP 2015-04-21: phase_option has been moved to the FidComputer
        #self.phase_option = 'PSU'
        # *** AJGP 2015-04-21: amp_update_mode has been replaced by tslot_type
        #self.amp_update_mode = 'ALL'  # Alts: 'DYNAMIC'
        self.fid_type = 'DEF'
        self.tslot_type = 'DEF'
        self.init_pulse_type = 'DEF'
        ######################
        # Note the following parameteres are for constrained optimisation
        # methods e.g. L-BFGS-B
        # *** AJGP 2015-04-21:
        #    These have been moved to the OptimizerLBFGSB class
        #        self.amp_lbound = -np.Inf
        #        self.amp_ubound = np.Inf
        #        self.max_metric_corr = 10
        #    These moved to termination conditions
        #        self.accuracy_factor = 1e7
        # ***
        # ####################

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    def check_create_output_dir(self, output_dir, desc='output'):
        """
        Checks if the given directory exists, if not it is created
        Returns
        -------
        dir_ok : boolean
            True if directory exists (previously or created)
            False if failed to create the directory

        output_dir : string
            Path to the directory, which may be been made absolute

        msg : string
            Error msg if directory creation failed
        """
        
        return qtrlio.create_dir(output_dir, desc=desc)

# create global instance
optimconfig = OptimConfig()

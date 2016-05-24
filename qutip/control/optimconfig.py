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
Configuration parameters for control pulse optimisation
"""

import numpy as np
# QuTiP logging
import qutip.logging_utils
logger = qutip.logging_utils.get_logger()
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

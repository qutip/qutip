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

import os
import numpy as np
# QuTiP logging
import qutip.logging
logger = qutip.logging.get_logger()

TEST_OUT_DIR = "test_out"


class OptimConfig:
    """
    Configuration parameters for control pulse optimisation

    Attributes
    ----------
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
        (See FideliyComputer classes for details)

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:
            PSU - global phase ignored
            SU - global phase included

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control
        (used in contrained methods only e.g. L-BFGS-B)

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control
        (used in contrained methods only e.g. L-BFGS-B)

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

    test_out_dir : string
        Directory where test output files will be saved
        By default this is a sub directory called 'test_out'
        It will be created in the working directory if it does not exist

    test_out_f_ext : string
        File extension that will be applied to all test output file names

    test_out_iter : Boolean
        When True a file will be created that records the wall time,
        fidelity error and gradient norm for each iteration of the algorithm

    test_out_fid_err : Boolean
        When True a file will be created that records the fidelity error
        each time the Optimizer.fid_err_wrapper method is called

    test_out_grad_norm : Boolean
        When True a file will be created that records the gradient norm
        each time the Optimizer.fid_err_grad_wrapper method is called

    test_out_grad : Boolean
        When True a file will be created each time the
        Optimizer.fid_err_grad_wrapper method is called containing
        the gradients with respect to each control in each timeslot

    test_out_prop : Boolean
        When True a file will be created each time the timeslot evolution
        is recomputed recording propagators for each timeslot

    test_out_prop_grad : Boolean
        When True a file will be created each time the timeslot evolution
        is recomputed recording the propagator gradient
        wrt each control in each timeslot

    test_out_evo : Boolean
        When True a file will be created each time the timeslot evolution
        is recomputed recording the operators (matrices) for the forward
        and onward evolution in each timeslot
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.log_level = logger.getEffectiveLevel()
        self.optim_alg = 'LBFGSB'
        self.dyn_type = ''
        self.fid_type = ''
        self.phase_option = 'PSU'
        self.amp_update_mode = 'ALL'  # Alts: 'DYNAMIC'
        self.pulse_type = 'RND'
        ######################
        # Note the following parameteres are for constrained optimisation
        # methods e.g. L-BFGS-B
        self.amp_lbound = -np.Inf
        self.amp_ubound = np.Inf
        self.max_metric_corr = 10
        self.accuracy_factor = 1e7
        # ####################
        self.reset_test_out_files()

    def reset_test_out_files(self):
        # Test output file flags
        self.test_out_dir = None
        self.test_out_f_ext = ".txt"
        self.test_out_iter = False
        self.test_out_fid_err = False
        self.test_out_grad_norm = False
        self.test_out_amps = False
        self.test_out_grad = False
        self.test_out_prop = False
        self.test_out_prop_grad = False
        self.test_out_evo = False

    def set_log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        self.log_level = lvl
        logger.setLevel(lvl)

    def any_test_files(self):
        """
        Returns True if any test_out_files are to be produced
        That is debug files written to the test_out directory
        """
        if (self.test_out_iter or
            self.test_out_fid_err or
            self.test_out_grad_norm or
            self.test_out_grad or
            self.test_out_amps or
            self.test_out_prop or
            self.test_out_prop_grad or
            self.test_out_evo):
            return True
        else:
            return False

    def check_create_test_out_dir(self):
        """
        Checks test_out folder exists, creates it if not
        """
        dir_ok = True
        self.test_out_dir = os.path.join(os.getcwd(), TEST_OUT_DIR)
        msg = "Failed to create test output file directory:\n{}\n".format(
            self.test_out_dir)
        if os.path.exists(self.test_out_dir):
            if os.path.isfile(self.test_out_dir):
                dir_ok = False
                msg += "A file already exists with same name"
        else:
            try:
                os.mkdir(TEST_OUT_DIR)
                logger.info("Test out files directory {} created".format(
                    TEST_OUT_DIR))
            except Exception as err:
                dir_ok = False
                msg += "Either turn off test_out_files or check permissions.\n"
                msg += "Underling error: {}".format(err)

        if not dir_ok:
            msg += "\ntest_out_files will be suppressed."
            logger.error(msg)
            self.test_out_files = 0

        return dir_ok

# create global instance
optimconfig = OptimConfig()

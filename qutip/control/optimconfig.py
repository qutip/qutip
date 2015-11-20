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
import errno
import numpy as np
# QuTiP logging
import qutip.logging_utils
logger = qutip.logging_utils.get_logger()

TEST_OUT_DIR = "test_out"


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
        (See FidelityComputer classes for details)

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
        #           These have been moved to the OptimizerLBFGSB class
#        self.amp_lbound = -np.Inf
#        self.amp_ubound = np.Inf
#        self.max_metric_corr = 10
#        self.accuracy_factor = 1e7
        # ***
        # ####################
        self.reset_test_out_files()

    def reset_test_out_files(self):
        # Test output file flags
        self.test_out_dir = None
        self.test_out_f_ext = ".txt"
        self.clear_test_out_flags()

    def clear_test_out_flags(self):
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
        Returns True if any test_out files are to be produced
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
        Checks test_out directory exists, creates it if not
        """
        if self.test_out_dir is None or len(self.test_out_dir) == 0:
            self.test_out_dir = TEST_OUT_DIR

        dir_ok, self.test_out_dir, msg = self.check_create_output_dir(
                    self.test_out_dir, desc='test_out')

        if not dir_ok:
            self.reset_test_out_files()
            msg += "\ntest_out files will be suppressed."
            logger.error(msg)

        return dir_ok

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

        dir_ok = True
        if '~' in output_dir:
            output_dir = os.path.expanduser(output_dir)
        elif not os.path.abspath(output_dir):
            # Assume relative path from cwd given
            output_dir = os.path.join(os.getcwd(), output_dir)

        errmsg = "Failed to create {} directory:\n{}\n".format(desc,
                                                            output_dir)

        if os.path.exists(output_dir):
            if os.path.isfile(output_dir):
                dir_ok = False
                errmsg += "A file already exists with the same name"
        else:
            try:
                os.makedirs(output_dir)
                logger.info("Test out files directory {} created "
                            "(recursively)".format(output_dir))
            except OSError as e:
                if e.errno == errno.EEXIST:
                    logger.info("Assume test out files directory {} created "
                        "(recursively)  some other process".format(output_dir))
                else:
                    dir_ok = False
                    errmsg += "Underling error (makedirs) :({}) {}".format(
                        type(e).__name__, e)

        if dir_ok:
            return dir_ok, output_dir, "{} directory is ready".format(desc)
        else:
            return dir_ok, output_dir, errmsg

# create global instance
optimconfig = OptimConfig()

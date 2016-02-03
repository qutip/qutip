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
Classes containing termination conditions for the control pulse optimisation
i.e. attributes that will be checked during the optimisation, that
will determine if the algorithm has completed its task / exceeded limits
"""


class TerminationConditions(object):
    """
    Base class for all termination conditions
    Used to determine when to stop the optimisation algorithm
    Note different subclasses should be used to match the type of
    optimisation being used

    Attributes
    ----------
    fid_err_targ : float
        Target fidelity error

    fid_goal : float
        goal fidelity, e.g. 1 - self.fid_err_targ
        It its typical to set this for unitary systems

    max_wall_time : float
        # maximum time for optimisation (seconds)

    min_gradient_norm : float
        Minimum normalised gradient after which optimisation will terminate

    max_iterations : integer
        Maximum iterations of the optimisation algorithm

    max_fid_func_calls : integer
        Maximum number of calls to the fidelity function during
        the optimisation algorithm
        
    accuracy_factor : float
        Determines the accuracy of the result.
        Typical values for accuracy_factor are: 1e12 for low accuracy;
        1e7 for moderate accuracy; 10.0 for extremely high accuracy
        scipy.optimize.fmin_l_bfgs_b factr argument.
        Only set for specific methods (fmin_l_bfgs_b) that uses this
        Otherwise the same thing is passed as method_option ftol
        (although the scale is different)
        Hence it is not defined here, but may be set by the user
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.fid_err_targ = 1e-5
        self.fid_goal = None
        self.max_wall_time = 60*60.0
        self.min_gradient_norm = 1e-5
        self.max_iterations = 1e10
        self.max_fid_func_calls = 1e10

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
Exception classes for the Quantum Control library
"""


class Error(Exception):
    """Base class for all qutip control exceptions"""

    def __str__(self):
        return repr(self.message)


class UsageError(Error):
    """
    A function has been used incorrectly. Most likely when a base class
    was used when a sub class should have been.
        funcname: function name where error occurred
        msg: Explanation
    
    """
    def __init__(self, msg):
        self.message = msg


class FunctionalError(Error):
    """
    A function behaved in an unexpected way
    Attributes:
        funcname: function name where error occurred
        msg: Explanation
    
    """
    def __init__(self, msg):
        self.message = msg


class OptimizationTerminate(Error):
    """
    Superclass for all early terminations from the optimisation algorithm
    
    """
    pass


class GoalAchievedTerminate(OptimizationTerminate):
    """
    Exception raised to terminate execution when the goal has been reached
    during the optimisation algorithm
    
    """
    def __init__(self, fid_err):
        self.reason = "Goal achieved"
        self.fid_err = fid_err


class MaxWallTimeTerminate(OptimizationTerminate):
    """
    Exception raised to terminate execution when the optimisation time has
    exceeded the maximum set in the config
    
    """
    def __init__(self):
        self.reason = "Max wall time exceeded"
        
class MaxFidFuncCallTerminate(OptimizationTerminate):
    """
    Exception raised to terminate execution when the number of calls to the 
    fidelity error function has exceeded the maximum
    
    """
    def __init__(self):
        self.reason = "Number of fidelity error calls has exceeded the maximum"

class GradMinReachedTerminate(OptimizationTerminate):
    """
    Exception raised to terminate execution when the minimum gradient normal
    has been reached during the optimisation algorithm
    
    """
    def __init__(self, gradient):
        self.reason = "Gradient normal minimum reached"
        self.gradient = gradient

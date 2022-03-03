# -*- coding: utf-8 -*-
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

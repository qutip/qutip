# -*- coding: utf-8 -*-
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


class TerminationConditions:
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

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 03 12:37:21 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Classes here are expected to implement a run_optimization function
that will use some method for optimising the control pulse, as defined
by the control amplitudes. The system that the pulse acts upon are defined
by the Dynamics object that must be passed in the instantiation.

The methods are typically N dimensional function optimisers that 
find the minima of a fidelity error function. Note the number of variables
for the fidelity function is the number of control timeslots, 
i.e. n_ctrls x Ntimeslots
The methods will call functions on the Dynamics.fid_computer object, 
one or many times per interation, 
to get the fidelity error and gradient wrt to the amplitudes.
The optimisation will stop when one of the termination conditions are met,
for example: the fidelity aim has be reached, a local minima has been found,
the maximum time allowed has been exceeded

These function optimisation methods are so far from SciPy.optimize
The two methods implemented are:
    BFGS - Broyden–Fletcher–Goldfarb–Shanno algorithm
        This a quasi second order Newton method. It uses successive calls to 
        the gradient function to make an estimation of the curvature (Hessian)
        and hence direct its search for the function minima
        The SciPy implementation is pure Python and hance is execution speed is
        not high
        use subclass: Optimizer_BFGS
        
    L-BFGS-B - Bounded, limited memory BFGS
        This a version of the BFGS method where the Hessian approximation is
        only based on a set of the most recent gradient calls. It generally
        performs better where the are a large number of variables
        The SciPy implementation of L-BFGS-B is wrapper around a well
        established and actively maintained implementation in Fortran
        Its is therefore very fast.
        # See SciPy documentation for credit and details on the 
        # scipy.optimize.fmin_l_bfgs_b function
        use subclass: Optimizer_LBFGSB

The baseclass Optimizer implements the function wrappers to the 
fidelity error, gradient, and iteration callback functions. 
These are called from the within the SciPy optimisation functions.
The subclasses implement the algorithm specific pulse optimisation function.

"""

import os
import numpy as np
import timeit
import scipy.optimize as spopt

import optimresult as optimresult
import termcond as termcond
import errors as errors
import utility as util
import dynamics as dynamics


# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class Optimizer:
    """
    Base class for all control pulse optimisers. This class should not be
    instantiated, use its subclasses
    This class implements the fidelity, gradient and interation callback
    functions.
    All subclass objects must be initialised with a 
        QtrlConfig object - various configuration options
        Dynamics object - describes the dynamics of the (quantum) system
                            to be control optimised
    """
    def __init__(self, config, dyn):
        self.dynamics = dyn
        self.config = config
        self.msg_level = config.msg_level
        self.test_out_files = self.config.test_out_files
        self.reset()
        
    def reset(self):
        self.termination_conditions = None
        self.pulse_generator = None
        self.num_iter = 0
        
        # This is the object used to collect stats for the optimisation
        # If it is not set, then stats are not callected, other than
        # those defined as properties of this object
        # Note it is (usually) shared with the dyn object
        self.stats = None
        self.wall_time_optim_start = 0.0

        
    def run_optimization(self, term_conds=None):
        """
        This class does not implement a method for running the optimisation
        A subclass should be used, e.g. Optimizer_LBFGSB
        
        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None
        """
        f = self.__class__.__name__ + ".run_optimization"
        m = "No method defined for running optimisation." + \
                " Suspect base class was used where sub class should have been"
        raise errors.UsageError(funcname=f, msg=m)

    def _create_result(self):
        """
        create the result object 
        and set the initial_amps attribute as the current amplitudes
        """
        result = optimresult.OptimResult()
        result.initial_amps = self.dynamics.ctrl_amps.copy()
        result.time = self.dynamics.time
        return result
        
    def _pre_run_check(self, term_conds):
        """
        Check optimiser attribute status and passed parameters before
        running the optimisation.
        """
        if (self.test_out_files >= 1 and self.stats == None):
            f = self.__class__.__name__ + "._pre_run_check"
            m = "Cannot output test files when stats object is not set"
            raise errors.UsageError(funcname=f, msg=m)
            
        if (term_conds is not None):
            self.termination_conditions = term_conds
        term_conds = self.termination_conditions
            
        if (not isinstance(term_conds, termcond.TerminationConditions)):
            f = self.__class__.__name__ + "._pre_run_check"
            m = "No termination conditions for the optimisation function"
            raise errors.UsageError(funcname=f, msg=m)
                
        if (not isinstance(self.dynamics, dynamics.Dynamics)):
            f = self.__class__.__name__ + "._pre_run_check"
            m = "No dynamics object attribute set"
            raise errors.UsageError(funcname=f, msg=m)
            
        if (term_conds.fid_err_targ == None and term_conds.fid_goal == None):
            f = self.__class__.__name__ + "._pre_run_check"
            m = "Either the goal or the fidelity error tolerance must be set"
            raise errors.UsageError(funcname=f, msg=m)
        
        if (term_conds.fid_err_targ == None):
            term_conds.fid_err_targ = np.abs(1 - term_conds.fid_goal)
        
        if (term_conds.fid_goal == None):
            term_conds.fid_goal = 1 - term_conds.fid_err_targ


    def fid_err_func_wrapper(self, *args):
        """
        Get the fidelity error achieved using the ctrl amplitudes passed
        in as the first argument.

        This is called by generic optimisation algorithm as the
        func to the minimised. The argument is the current 
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)
        
        The error is checked against the target, and the optimisation is
        terminated if the target has been achieved.
        """
        # *** update stats ***
        if (self.stats != None):
            self.stats.num_fidelity_func_calls += 1
            if (self.msg_level > 0):
                print "Computing fidelity error {}".\
                        format(self.stats.num_fidelity_func_calls)
        amps = args[0].copy().reshape(self.dynamics.ctrl_amps.shape)
        self.dynamics.update_ctrl_amps(amps)
        
        tc = self.termination_conditions
        err = self.dynamics.fid_computer.get_fid_err()
        if (err <= tc.fid_err_targ):
            raise errors.GoalAchievedTerminate(err)
        
        return err
        
    def fid_err_grad_wrapper(self, *args):
        """
        Get the gradient of the fidelity error with respect to all of the
        variables, i.e. the ctrl amplidutes in each timeslot

        This is called by generic optimisation algorithm as the gradients of
        func to the minimised wrt the variables. The argument is the current 
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)
        
        Although the optimisation algorithms have a check within them for
        function convergence, i.e. local minima, the sum of the squares
        of the normalised gradient is checked explicitly, and the 
        optimisation is terminated if this is below the min_gradient_norm
        condition
        """
        # *** update stats ***
        if (self.stats != None):
            self.stats.num_grad_func_calls += 1
            if (self.msg_level > 0):
                print "Computing gradient normal {}".\
                        format(self.stats.num_grad_func_calls)
        amps = args[0].copy().reshape(self.dynamics.ctrl_amps.shape)
        self.dynamics.update_ctrl_amps(amps)
        fidComp = self.dynamics.fid_computer
        # gradient_norm_func is a pointer to the function set in the config
        # that returns the normalised gradients
        grad = fidComp.get_fid_err_gradient()
        if (self.test_out_files >= 1):
            # save gradients to file
            fname = os.path.join("test_out", \
                    "grad_{}_{}_call{}.txt".format(self.config.dyn_type, \
                            self.config.fid_type, \
                            self.stats.num_grad_func_calls))
            util.write_array_to_file(grad, fname, dtype=float)
        
        tc = self.termination_conditions
        if (fidComp.norm_grad_sq_sum < tc.min_gradient_norm):
            raise errors.GradMinReachedTerminate(fidComp.norm_grad_sq_sum)
        return grad.flatten()
        
    def iter_step_callback_func(self, *args):
        """
        Check the elapsed wall time for the optimisation run so far.
        Terminate if this has exceeded the maximum allowed time
        """
        if (self.msg_level > 0):
            print "Iteration callback {}".format(self.num_iter)
            
        tc = self.termination_conditions
        if (timeit.default_timer() - self.wall_time_optimize_start > 
                tc.max_wall_time):
            raise errors.MaxWallTimeTerminate()
                
        self.num_iter += 1
        # *** update stats ***
        if (self.stats != None):
            self.stats.num_iter = self.num_iter
            
    def _interpret_term_exception(self, except_term, result):
        """
        Update the result object based on the exception that occurred
        during the optimisation
        """
        result.termination_reason = except_term.reason
        if (isinstance(except_term, errors.GoalAchievedTerminate)):
            result.goal_achieved = True
        elif (isinstance(except_term, errors.MaxWallTimeTerminate)):
            result.wall_time_limit_exceeded = True
        elif (isinstance(except_term, errors.GradMinReachedTerminate)):
            result.grad_norm_min_reached = True
            
    def _add_common_result_attribs(self, result, st_time, end_time):
        """
        Update the result object attributes which are common to all 
        optimisers and outcomes
        """
        dyn = self.dynamics
        result.num_iter = self.num_iter
        result.wall_time = end_time - st_time
        result.fid_err = dyn.fid_computer.get_fid_err()
        result.grad_norm_final = dyn.fid_computer.norm_grad_sq_sum
        result.final_amps = dyn.ctrl_amps
        result.evo_full_final = dyn.Evo_init2t[dyn.num_tslots]
        # *** update stats ***
        if (self.stats is not None):
            self.stats.wall_time_optim_end = end_time
            self.stats.calculate()
            result.stats = self.stats
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]    

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class Optimizer_BFGS(Optimizer):
    """
    Implements the run_optimization method using the BFGS algorithm
    """
    
    def run_optimization(self, term_conds=None):
        """
        Optimise the control pulse amplitudes to minimise the fidelity error
        using the BFGS (Broyden–Fletcher–Goldfarb–Shanno) algorithm
        The optimisation end when one of the passed termination conditions 
        has been met, e.g. target achieved, gradient minimum met 
        (local minima), wall time / iteration count exceeded.
        
        Essentially this is wrapper to the:
        scipy.optimize.fmin_bfgs
        function
        
        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None
        
        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc
        """
        self._pre_run_check(term_conds)
        term_conds = self.termination_conditions
        dyn = self.dynamics
        x0 = dyn.ctrl_amps.reshape([-1])
        self.num_iter = 1
        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time
        
        if (self.stats != None):
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 1
        
        if (self.msg_level > 0):
            print "Optimising using BFGS"
        result = self._create_result()
        try:
            amps, cost, grad, invHess, nFCalls, nGCalls, warn = \
                spopt.fmin_bfgs(self.fid_err_func_wrapper, x0, \
                                fprime=self.fid_err_grad_wrapper, \
                                callback=self.iter_step_callback_func, \
                                gtol=term_conds.min_gradient_norm, \
                                maxiter=term_conds.max_iterations, \
                                full_output=True, disp=True)
        
            amps = amps.reshape([dyn.num_tslots, self.config.num_ctrls])
            dyn.update_ctrl_amps(amps)
            if (warn == 1):
                result.max_iter_exceeded = True
                result.termination_reason = "Iteration count limit reached"
            elif (warn == 2):
                result.grad_static = True
                result.termination_reason = "Gradient normal minimum reached"
            
        except errors.OptimisationTerminate as except_term:
            self._interpret_term_exception(except_term, result)
        
        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)
        
        return result
        
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]    

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[

class Optimizer_LBFGSB(Optimizer):
    def _build_bounds_list(self):
        cfg = self.config
        dyn = self.dynamics
        n_ctrls = dyn.get_num_ctrls()
        bounds = []
        for t in range(dyn.num_tslots):
            for c in range(n_ctrls):
                if (isinstance(cfg.amp_lbound, list)):
                    lb = cfg.amp_lbound[c]
                else:
                    lb = cfg.amp_lbound
                if (isinstance(cfg.amp_ubound, list)):
                    ub = cfg.amp_ubound[c]
                else:
                    ub = cfg.amp_ubound
                bounds.append((lb, ub))
                
        return bounds
                
    def run_optimization(self, term_conds=None):
        """
        Optimise the control pulse amplitudes to minimise the fidelity error
        using the L-BFGS-B algorithm, which is the constrained 
        (bounded amplitude values), limited memory, version of the
        Broyden–Fletcher–Goldfarb–Shanno algorithm.
        
        The optimisation end when one of the passed termination conditions 
        has been met, e.g. target achieved, gradient minimum met 
        (local minima), wall time / iteration count exceeded.
        
        Essentially this is wrapper to the:
        scipy.optimize.fmin_l_bfgs_b function
        This in turn is a warpper for well established implementation of
        the L-BFGS-B algorithm written in Fortran, which is therefore
        very fast. See SciPy documentation for credit and details on 
        this function.
        
        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None
        
        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc
        """
        self._pre_run_check(term_conds)
        term_conds = self.termination_conditions
        dyn = self.dynamics
        cfg = self.config
        x0 = dyn.ctrl_amps.reshape([-1])
        self.num_iter = 1
        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time
        
        if (self.stats != None):
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 1
        
        bounds = self._build_bounds_list();
        result = self._create_result()
        if (self.msg_level > 0):
            print "Optimising using L-BFGS-B"
        try:
            amps, fid, res_dict = \
                spopt.fmin_l_bfgs_b(self.fid_err_func_wrapper, x0, \
                                fprime=self.fid_err_grad_wrapper, \
                                callback=self.iter_step_callback_func, \
                                bounds=bounds, \
                                m=cfg.max_metric_corr, \
                                factr=cfg.optim_alg_acc_fact, \
                                pgtol=term_conds.min_gradient_norm, \
                                iprint=self.msg_level - 1, \
                                maxfun=term_conds.max_fid_func_calls, \
                                maxiter=term_conds.max_iterations)
        
            amps = amps.reshape([dyn.num_tslots, dyn.num_ctrls])
            dyn.update_ctrl_amps(amps)
            warn = res_dict['warnflag']
            if (warn == 0):
                result.grad_static = True
                result.termination_reason = "function converged"
            elif (warn == 1):
                result.max_iter_exceeded = True
                result.termination_reason = \
                    "Iteration or fidelity function call limit reached"
            elif (warn == 2):
                result.termination_reason = res_dict['task']

            result.num_iter = res_dict['nit']           
        except errors.OptimisationTerminate as except_term:
            self._interpret_term_exception(except_term, result)
        
        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)
        
        return result
    
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]    


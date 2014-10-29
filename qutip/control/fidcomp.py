# -*- coding: utf-8 -*-
"""
Created on Wed Sep 03 11:40:17 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Fidelity Computer

These classes calculate the fidelity error - function to be minimised
and fidelity error gradient, which is used to direct the optimisation

They may calculate the fidelity as an intermediary step, as in some case
e.g. unitary dynamics, this is more efficient

The idea is that different methods for computing the fidelity can be tried
and compared using simple configuration switches.

Note the methods in these classes were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
The unitary dynamics fidelity is taken directly frm DYNAMO
The other fidelity measures are extensions, and the sources are given
in the class descriptions.
"""

import os
import numpy as np
#import scipy.linalg as la
import timeit
import errors as errors
import utility as util


# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class FideliyComputer:
    """
    Base class for all Fidelity Computers. 
    This cannot be used directly. See subclass descriptions and choose
    one appropriate for the application
    Note: this must be instantiated with a Dynamics object, that is the
    container for the data that the methods operate on
    """
    def __init__(self, dynamics):
        self.parent = dynamics
        self.reset()
        
    def reset(self):
        """
        reset any configuration data and
        clear any temporarily held status data
        """
        self.msg_level = self.parent.msg_level
        self.dimensional_norm = 1.0
        self.fid_norm_func = None
        self.grad_norm_func = None
        self.uses_evo_t2end = False
        self.uses_evo_t2targ = False
        self.clear()
    
    def clear(self):
        """
        clear any temporarily held status data
        """
        self.fid_err = None
        self.fidelity = None
        self.fid_err_grad = None
        self.norm_grad_sq_sum = np.inf
        self.fidelity_current = False
        self.fid_err_grad_current = False
        self.norm_grad_sq_sum = 0.0
                
    def init_comp(self):
        """
        initialises the computer based on the configuration of the Dynamics
        """
        # optionally implemented in subclass
        pass
    
    def get_fid_err(self):
        """
        returns the absolute distance from the maximum achievable fidelity
        """
        # must be implemented by subclass
        f = self.__class__.__name__ + ".get_fid_err"
        m = "No method defined for getting fidelity error." + \
                " Suspect base class was used where sub class should have been"
        raise errors.UsageError(funcname=f, msg=m)
    
    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x nCtrls) array wrt the timeslot control amplitude
        """
        # must be implemented by subclass
        f = self.__class__.__name__ + ".get_fid_err_gradient"
        m = "No method defined for getting fidelity error gradient." + \
                " Suspect base class was used where sub class should have been"
        raise errors.UsageError(funcname=f, msg=m)
        
    def flag_system_changed(self):
        """
        Flag fidelity and gradients as needing recalculation
        """
        self.fidelity_current = False
        # Flag gradient as needing recalculating
        self.fid_err_grad_current = False
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[

class FidComp_Unitary(FideliyComputer):
    """
    Computes fidelity error and gradient assuming unitary dynamics, e.g.
    closed qubit systems
    Note fidelity and gradient calculations were taken from DYNAMO 
    (see file header)
    """
    
    def reset(self):
        FideliyComputer.reset(self)
        self.uses_evo_t2targ = True
    
    def clear(self):
        FideliyComputer.clear(self)
        self.fidelity_prenorm = None
        self.fidelity_prenorm_current = False
    
    def set_phase_option(self, phase_option='PSU'):
        """
        # Phase options are 
        #  SU - global phase important
        #  PSU - global phase is not important
        """
        if (phase_option == 'PSU'):
            self.fid_norm_func = self.normalize_PSU
            self.grad_norm_func = self.normalize_gradient_PSU
        elif (phase_option == 'SU'):
            self.fid_norm_func = self.normalize_SU
            self.grad_norm_func = self.normalize_gradient_SU
        elif (phase_option is None):
            f = self.__class__.__name__ + ".init_comp"
            m = 'phase_option cannot be set to None for this FidelityComputer.'
            raise errors.UsageError(funcname=f, msg=m)
        else:
            f = self.__class__.__name__ + ".init_comp"
            m = 'No option for cfg.phase_option: ' + phase_option
            raise errors.UsageError(funcname=f, msg=m)

    def init_comp(self):
        """
        Check configuration and initialise the normalisation
        """
        if (self.fid_norm_func is None or self.grad_norm_func is None):
            f = self.__class__.__name__ + ".init_comp"
            m = "The phase option must be be set for this fidelity computer"
            raise errors.UsageError(funcname=f, msg=m)
        self.init_normalization()
        
    def flag_system_changed(self):
        """
        Flag fidelity and gradients as needing recalculation
        """        
        FideliyComputer.flag_system_changed(self)
        # Flag the fidelity (prenormalisation) value as needing calculation
        self.fidelity_prenorm_current = False

    #####################################
    # Normalisation functions
    #####################################
    def init_normalization(self):
        """
        Calc norm of <Ufinal | Ufinal> to scale subsequent norms
        When considering unitary time evolution operators, this basically 
        results in calculating the trace of the identity matrix
        and is hence equal to the size of the target matrix
        There may be situations where this is not the case, and hence it
        is not assumed to be so.
        The normalisation function called should be set to either the
        PSU - global phase ignored
        SU  - global phase respected
        """
        dyn = self.parent
        self.dimensional_norm = 1.0
        self.dimensional_norm = \
            self.fid_norm_func(dyn.target.conj().T.dot(dyn.target))
            
    def normalize_SU(self, A):
        """
        
        """
        if (isinstance(A, np.ndarray)):
            #input is an array (matrix), so 
            norm =  np.trace(A)
        else:
            #input is already scalar and hence assumed
            # to be the prenormalised scalar value, e.g. fidelity
            norm =  A
        return np.real(norm) / self.dimensional_norm
        
    def normalize_gradient_SU(self, grad):
        """
        Normalise the gradient matrix passed as grad
        This SU version respects global phase
        """
        grad_normalized = np.real(grad) / self.dimensional_norm
            
        return grad_normalized
        
    def normalize_PSU(self, A):
        """
        
        """
        if (isinstance(A, np.ndarray)):
            #input is an array (matrix), so 
            norm =  np.trace(A)
        else:
            #input is already scalar and hence assumed
            # to be the prenormalised scalar value, e.g. fidelity
            norm =  A
        return np.abs(norm) / self.dimensional_norm
        
    def normalize_gradient_PSU(self, grad):
        """
        Normalise the gradient matrix passed as grad
        This PSU version is independent of global phase
        """
        fidelityPrenorm = self.get_fidelity_prenorm()
        grad_normalized = \
            2*np.real(grad*np.conj(fidelityPrenorm)) / self.dimensional_norm
        return grad_normalized

    ######################################################
    ### Fidelity (Figure of merit functions) #############
    ######################################################

    def get_fid_err(self):
        """
        Gets the absolute error in the fidelity
        """
        return np.abs(1 - self.get_fidelity())
        
    def get_fidelity(self):
        """
        Gets the appropriately normalised fidelity value
        The normalisation is determined by the fid_norm_func pointer
        which should be set in the config
        """
        if (not self.fidelity_current):
            self.fidelity = \
                self.fid_norm_func(self.get_fidelity_prenorm())
            self.fidelity_current = True
            if (self.msg_level >= 2):
                print "Fidelity (normalised): " + \
                        str(self.fidelity) 

        return self.fidelity
        
    def get_fidelity_prenorm(self):
        """
        Gets the current fidelity value prior to normalisation
        Note the gradient function uses this value
        The value is cached, because it is used in the gradient calculation
        """
        if (not self.fidelity_prenorm_current):
            dyn = self.parent
            k = dyn.tslot_computer.get_timeslot_for_fidelity_calc()
            dyn.compute_evolution()
            f = np.trace(dyn.Evo_init2t[k].dot(dyn.Evo_t2targ[k]))
            self.fidelity_prenorm = f
            self.fidelity_prenorm_current = True
            if (dyn.stats != None):
                    dyn.stats.num_fidelity_computes += 1
            if (self.msg_level >= 2):
                print "Fidelity (pre normalisation): " + \
                        str(self.fidelity_prenorm) 
        return self.fidelity_prenorm
            
    ########################################
    ## Gradient functions
    ########################################
        
    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x nCtrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates 
        (although this is not typically found to happen)
        """
        if (not self.fid_err_grad_current):
            dyn = self.parent
            gradPreNorm = self.compute_fid_grad()
            if (self.msg_level >= 5):
                print "pre-normalised fidelity gradients:"
                print gradPreNorm
            # AJGP: Note this check should not be necessary if dynamics are
            #       unitary. However, if they are not then this gradient
            #       can still be used, however the interpretation is dubious
            if (self.get_fidelity() >= 1):
                self.fid_err_grad = self.grad_norm_func(gradPreNorm)
            else:
                self.fid_err_grad = -self.grad_norm_func(gradPreNorm)
                
            self.fid_err_grad_current = True
            if (dyn.stats != None):
                dyn.stats.num_grad_computes += 1

            self.norm_grad_sq_sum = np.sum(self.fid_err_grad**2)
            if (self.msg_level >= 4):
                print "Normalised fidelity error gradients:"
                print self.fid_err_grad
            if (self.msg_level >= 2):
                print "Grad (sum sq norm): " + \
                        str(self.norm_grad_sq_sum)
                
        return self.fid_err_grad
        
    def compute_fid_grad(self):
        """
        Calculates exact gradient of function wrt to each timeslot 
        control amplitudes. Note these gradients are not normalised
        These are returned as a (nTimeslots x nCtrls) array
        """
        dyn = self.parent
        nCtrls = dyn.get_num_ctrls()
        nTS = dyn.num_tslots
        
        # create nTS x nCtrls zero array for grad start point
        grad = np.zeros([nTS, nCtrls], dtype=complex)
        
        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()
        
        # loop through all ctrl timeslots calculating gradients
        timeStart = timeit.default_timer()
        for j in range(nCtrls):
            for k in range(nTS):
                owdEvo = dyn.Evo_t2targ[k+1]
                fwdEvo = dyn.Evo_init2t[k]
                if (self.msg_level >= 5):
                    fname = os.path.join("test_out", \
                                "prop_grad_UNIT_j{}_k{}.txt".format(\
                                j, k))
                    util.write_array_to_file(dyn.Prop_grad[k, j], \
                                                       fname, dtype=complex)
                
                g = np.trace(owdEvo.dot(dyn.Prop_grad[k, j]).dot(fwdEvo))
                grad[k, j] = g
        if (dyn.stats != None):
            dyn.stats.wall_time_gradient_compute += \
                    timeit.default_timer() - timeStart
        #import sys
        #sys.exit("Early exit. Checking gradients")
        return grad
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class FidComp_TraceDiff(FideliyComputer):
    """
    Computes fidelity error and gradient for general system dynamics
    by calculating the the fidelity error as the trace of the overlap
    of the difference between the target and evolution resulting from
    the pulses with the transpose of the same.
    This should provide a distance measure for dynamics described by matrices
    Note the gradient calculation is taken from:
    'Robust quantum gates for open systems via optimal control: 
    Markovian versus non-Markovian dynamics'
    Frederik F Floether, Pierre de Fouquieres, and Sophie G Schirmer
    """
    
    def reset(self):
        FideliyComputer.reset(self)
        self.scale_factor = 1.0
        self.uses_evo_t2end = True
        if (not self.parent.prop_computer.grad_exact):
            f = self.__class__.__name__ + ".reset"
            m = "This FideliyComputer can only be used with an exact " + \
                    "gradient PropagatorComputer."
            raise errors.UsageError(funcname=f, msg=m)   
        
    def get_fid_err(self):
        """
        Gets the absolute error in the fidelity
        """
        if (not self.fidelity_current):
            dyn = self.parent
            dyn.compute_evolution()
            nTS = dyn.num_tslots
            evoFinal = dyn.Evo_init2t[nTS]
            evoFinalDiff = dyn.target - evoFinal
            if (self.msg_level >= 4):
                print "Target:"
                print dyn.target
                print "evo final:"
                print evoFinal
                print "evo final diff:"
                print evoFinalDiff
                
            # Calculate the fidelity error using the trace difference norm
            # Note that the value should have not imagnary part, so using
            # np.real, just avoids the complex casting warning
            self.fid_err = self.scale_factor*np.real(\
                        np.trace(evoFinalDiff.conj().T.dot(evoFinalDiff)))
            if (dyn.stats != None):
                    dyn.stats.num_fidelity_computes += 1
                    
            self.fidelity_current = True
            if (self.msg_level >= 2):
                print "Fidelity error: " + \
                        str(self.fid_err) 

        return self.fid_err
        
    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x nCtrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates 
        (although this is not typically found to happen)
        """
        if (not self.fid_err_grad_current):
            dyn = self.parent
            self.fid_err_grad = self.compute_fid_err_grad() #\
                               #/ self.dimensional_norm
                
            self.fid_err_grad_current = True
            if (dyn != None):
                dyn.stats.num_grad_computes += 1

            self.norm_grad_sq_sum = np.sum(self.fid_err_grad**2)
            if (self.msg_level >= 4):
                print "Normalised fidelity error gradients:"
                print self.fid_err_grad
            if (self.msg_level >= 2):
                print "Grad (sum sq norm): " + \
                        str(self.norm_grad_sq_sum)
                
        return self.fid_err_grad
        
    def compute_fid_err_grad(self):
        """
        Calculate exact gradient of the fidelity error function 
        wrt to each timeslot control amplitudes. 
        Uses the trace difference norm fidelity
        These are returned as a (nTimeslots x nCtrls) array
        """
        dyn = self.parent
        nCtrls = dyn.get_num_ctrls()
        nTS = dyn.num_tslots
        
        # create nTS x nCtrls zero array for grad start point
        grad = np.zeros([nTS, nCtrls])
        
        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()
        
        # loop through all ctrl timeslots calculating gradients
        timeStart = timeit.default_timer()
        evoFinal = dyn.Evo_init2t[nTS]
        evoFinalDiff = dyn.target - evoFinal
        
        for j in range(nCtrls):
            for k in range(nTS):
                fwdEvo = dyn.Evo_init2t[k]
                evoGrad = dyn.Prop_grad[k, j].dot(fwdEvo)
                
                if (k + 1 < nTS):
                    owdEvo = dyn.Evo_t2end[k+1]
                    evoGrad = owdEvo.dot(evoGrad)
                    
                # Note that the value should have not imagnary part, so using
                # np.real, just avoids the complex casting warning                    
                g = -2*self.scale_factor*np.real( \
                        np.trace(evoFinalDiff.conj().T.dot(evoGrad)))
                grad[k, j] = g
        if (dyn != None):
            dyn.stats.wall_time_gradient_compute += \
                    timeit.default_timer() - timeStart
        #import sys
        #sys.exit("Early exit. Checking gradients")
        return grad
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

        

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
# AJGP 2014-10-02
# Added as new class
class FidComp_TraceDiff_ApproxGrad(FidComp_TraceDiff):
    """
    As FidComp_TraceDiff, except uses the finite difference method to 
    compute approximate gradients
    """
    def reset(self):
        FideliyComputer.reset(self)
        self.uses_evo_t2end = True
        self.scale_factor = 1.0
        self.epsilon = 0.001
        
    def compute_fid_err_grad(self):
        """
        Calculates gradient of function wrt to each timeslot 
        control amplitudes. Note these gradients are not normalised
        They are calulated 
        These are returned as a (nTimeslots x nCtrls) array
        """
        dyn = self.parent
        prop_comp = dyn.prop_computer
        nCtrls = dyn.get_num_ctrls()
        nTS = dyn.num_tslots
        
        if (self.msg_level >= 2):
            print "Computing fidelity error gradient using " + \
                        self.__class__.__name__
        # create nTS x nCtrls zero array for grad start point
        grad = np.zeros([nTS, nCtrls])
        
        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()
        currFidErr = self.get_fid_err()
        
        # loop through all ctrl timeslots calculating gradients
        timeStart = timeit.default_timer()
        
        for j in range(nCtrls):
            for k in range(nTS):
                fwdEvo = dyn.Evo_init2t[k]
                propEps = prop_comp.compute_diff_prop(k, j, self.epsilon)
                evoFinalEps = fwdEvo.dot(propEps)
                if (k + 1 < nTS):
                    owdEvo = dyn.Evo_t2end[k+1]
                    evoFinalEps = evoFinalEps.dot(owdEvo)
                evoFinalDiffEps = dyn.target - evoFinalEps
                # Note that the value should have not imagnary part, so using
                # np.real, just avoids the complex casting warning                    
                fidErrEps = self.scale_factor*np.real( \
                            np.trace(evoFinalDiffEps.T.dot(evoFinalDiffEps)))
                g = (fidErrEps - currFidErr)/self.epsilon
                        
                grad[k, j] = g
        if (dyn != None):
            dyn.stats.wall_time_gradient_compute += \
                    timeit.default_timer() - timeStart
        #import sys
        #sys.exit("Early exit. Checking gradients")
        return grad
    
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
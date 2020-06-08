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
import warnings
import numpy as np
import scipy.sparse as sp
# import scipy.linalg as la
import timeit
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors

warnings.simplefilter('always', DeprecationWarning) #turn off filter
def _attrib_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

def _func_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    
def _trace(A):
    """wrapper for calculating the trace"""
    # input is an operator (Qobj, array, sparse etc), so
    if isinstance(A, Qobj):
        return A.tr()
    elif isinstance(A, np.ndarray):
        return np.trace(A)
    else:
        #Assume A some sparse matrix
        return np.sum(A.diagonal())


class FidelityComputer(object):
    """
    Base class for all Fidelity Computers.
    This cannot be used directly. See subclass descriptions and choose
    one appropriate for the application
    Note: this must be instantiated with a Dynamics object, that is the
    container for the data that the methods operate on

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

    dimensional_norm : float
        Normalisation constant

    fid_norm_func : function
        Used to normalise the fidelity
        See SU and PSU options for the unitary dynamics

    grad_norm_func : function
        Used to normalise the fidelity gradient
        See SU and PSU options for the unitary dynamics

    uses_onwd_evo : boolean
        flag to specify whether the onwd_evo evolution operator
        (see Dynamics) is used by the FidelityComputer

    uses_onto_evo : boolean
        flag to specify whether the onto_evo evolution operator
         (see Dynamics) is used by the FidelityComputer

    fid_err : float
        Last computed value of the fidelity error

    fidelity : float
        Last computed value of the normalised fidelity

    fidelity_current : boolean
        flag to specify whether the fidelity / fid_err are based on the
        current amplitude values. Set False when amplitudes change

    fid_err_grad: array[num_tslot, num_ctrls] of float
        Last computed values for the fidelity error gradients wrt the
        control in the timeslot

    grad_norm : float
        Last computed value for the norm of the fidelity error gradients
        (sqrt of the sum of the squares)

    fid_err_grad_current : boolean
        flag to specify whether the fidelity / fid_err are based on the
        current amplitude values. Set False when amplitudes change
    """
    def __init__(self, dynamics, params=None):
        self.parent = dynamics
        self.params = params
        self.reset()

    def reset(self):
        """
        reset any configuration data and
        clear any temporarily held status data
        """
        self.log_level = self.parent.log_level
        self.id_text = 'FID_COMP_BASE'
        self.dimensional_norm = 1.0
        self.fid_norm_func = None
        self.grad_norm_func = None
        self.uses_onwd_evo = False
        self.uses_onto_evo = False
        self.apply_params()
        self.clear()

    def clear(self):
        """
        clear any temporarily held status data
        """
        self.fid_err = None
        self.fidelity = None
        self.fid_err_grad = None
        self.grad_norm = np.inf
        self.fidelity_current = False
        self.fid_err_grad_current = False
        self.grad_norm = 0.0

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

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
        raise errors.UsageError(
            "No method defined for getting fidelity error."
            " Suspect base class was used where sub class should have been")

    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array wrt the timeslot control amplitude
        """
        # must be implemented by subclass
        raise errors.UsageError("No method defined for getting fidelity"
                                " error gradient. Suspect base class was"
                                " used where sub class should have been")

    def flag_system_changed(self):
        """
        Flag fidelity and gradients as needing recalculation
        """
        self.fidelity_current = False
        # Flag gradient as needing recalculating
        self.fid_err_grad_current = False

    @property
    def uses_evo_t2end(self):
        _attrib_deprecation(
                "'uses_evo_t2end' has been replaced by 'uses_onwd_evo'")
        return self.uses_onwd_evo

    @uses_evo_t2end.setter
    def uses_evo_t2end(self, value):
        _attrib_deprecation(
                "'uses_evo_t2end' has been replaced by 'uses_onwd_evo'")
        self.uses_onwd_evo = value

    @property
    def uses_evo_t2targ(self):
        _attrib_deprecation(
                "'uses_evo_t2targ' has been replaced by 'uses_onto_evo'")
        return self.uses_onto_evo

    @uses_evo_t2targ.setter
    def uses_evo_t2targ(self, value):
        _attrib_deprecation(
                "'uses_evo_t2targ' has been replaced by 'uses_onto_evo'")
        self.uses_onto_evo = value

class FidCompUnitary(FidelityComputer):
    """
    Computes fidelity error and gradient assuming unitary dynamics, e.g.
    closed qubit systems
    Note fidelity and gradient calculations were taken from DYNAMO
    (see file header)

    Attributes
    ----------
    phase_option : string
        determines how global phase is treated in fidelity calculations:
            PSU - global phase ignored
            SU - global phase included

    fidelity_prenorm : complex
        Last computed value of the fidelity before it is normalised
        It is stored to use in the gradient normalisation calculation

    fidelity_prenorm_current : boolean
        flag to specify whether fidelity_prenorm are based on the
        current amplitude values. Set False when amplitudes change
    """

    def reset(self):
        FidelityComputer.reset(self)
        self.id_text = 'UNIT'
        self.uses_onto_evo = True
        self._init_phase_option('PSU')
        self.apply_params()

    def clear(self):
        FidelityComputer.clear(self)
        self.fidelity_prenorm = None
        self.fidelity_prenorm_current = False

    def set_phase_option(self, phase_option=None):
        """
        Deprecated - use phase_option
        Phase options are
        SU - global phase important
        PSU - global phase is not important
        """
        _func_deprecation("'set_phase_option' is deprecated. "
                            "Use phase_option property")
        self._init_phase_option(phase_option)
        
    @property
    def phase_option(self):
        return self._phase_option
        
    @phase_option.setter
    def phase_option(self, value):
        """
        # Phase options are
        #  SU - global phase important
        #  PSU - global phase is not important
        """
        self._init_phase_option(value)

    def _init_phase_option(self, value):
        self._phase_option = value
        if value == 'PSU':
            self.fid_norm_func = self.normalize_PSU
            self.grad_norm_func = self.normalize_gradient_PSU
        elif value == 'SU':
            self.fid_norm_func = self.normalize_SU
            self.grad_norm_func = self.normalize_gradient_SU
        elif value is None:
            raise errors.UsageError("phase_option cannot be set to None"
                                    " for this FidelityComputer.")
        else:
            raise errors.UsageError(
                    "No option for phase_option '{}'".format(value))
                                                            
    def init_comp(self):
        """
        Check configuration and initialise the normalisation
        """
        if self.fid_norm_func is None or self.grad_norm_func is None:
            raise errors.UsageError("The phase_option must be be set"
                                    "for this fidelity computer")
        self.init_normalization()

    def flag_system_changed(self):
        """
        Flag fidelity and gradients as needing recalculation
        """
        FidelityComputer.flag_system_changed(self)
        # Flag the fidelity (prenormalisation) value as needing calculation
        self.fidelity_prenorm_current = False


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
            self.fid_norm_func(dyn.target.dag()*dyn.target)

    def normalize_SU(self, A):
        """

        """
        try:
            if A.shape[0] == A.shape[1]:
               # input is an operator (Qobj, array, sparse etc), so
                norm = _trace(A)
            else:
                raise TypeError("Cannot compute trace (not square)")
        except:
            # assume input is already scalar and hence assumed
            # to be the prenormalised scalar value, e.g. fidelity
            norm = A
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
        try:
            if A.shape[0] == A.shape[1]:
               # input is an operator (Qobj, array, sparse etc), so
                norm = _trace(A)
            else:
                raise TypeError("Cannot compute trace (not square)")
        except:
            # assume input is already scalar and hence assumed
            # to be the prenormalised scalar value, e.g. fidelity
            norm = A
        return np.abs(norm) / self.dimensional_norm

    def normalize_gradient_PSU(self, grad):
        """
        Normalise the gradient matrix passed as grad
        This PSU version is independent of global phase
        """
        fid_pn = self.get_fidelity_prenorm()
        grad_normalized = np.real(grad * np.exp(-1j * np.angle(fid_pn)) /
                                  self.dimensional_norm)
        return grad_normalized

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
        if not self.fidelity_current:
            self.fidelity = \
                self.fid_norm_func(self.get_fidelity_prenorm())
            self.fidelity_current = True
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity (normalised): {}".format(self.fidelity))

        return self.fidelity

    def get_fidelity_prenorm(self):
        """
        Gets the current fidelity value prior to normalisation
        Note the gradient function uses this value
        The value is cached, because it is used in the gradient calculation
        """
        if not self.fidelity_prenorm_current:
            dyn = self.parent
            k = dyn.tslot_computer._get_timeslot_for_fidelity_calc()
            dyn.compute_evolution()
            if dyn.oper_dtype == Qobj:
                f = (dyn._onto_evo[k]*dyn._fwd_evo[k]).tr()
            else:
                f = _trace(dyn._onto_evo[k].dot(dyn._fwd_evo[k]))
            self.fidelity_prenorm = f
            self.fidelity_prenorm_current = True
            if dyn.stats is not None:
                    dyn.stats.num_fidelity_computes += 1
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity (pre normalisation): {}".format(
                    self.fidelity_prenorm))
        return self.fidelity_prenorm

    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates
        (although this is not typically found to happen)
        """
        if not self.fid_err_grad_current:
            dyn = self.parent
            grad_prenorm = self.compute_fid_grad()
            if self.log_level <= logging.DEBUG_INTENSE:
                logger.log(logging.DEBUG_INTENSE, "pre-normalised fidelity "
                           "gradients:\n{}".format(grad_prenorm))
            # AJGP: Note this check should not be necessary if dynamics are
            #       unitary. However, if they are not then this gradient
            #       can still be used, however the interpretation is dubious
            if self.get_fidelity() >= 1:
                self.fid_err_grad = self.grad_norm_func(grad_prenorm)
            else:
                self.fid_err_grad = -self.grad_norm_func(grad_prenorm)

            self.fid_err_grad_current = True
            if dyn.stats is not None:
                dyn.stats.num_grad_computes += 1

            self.grad_norm = np.sqrt(np.sum(self.fid_err_grad**2))
            if self.log_level <= logging.DEBUG_INTENSE:
                logger.log(logging.DEBUG_INTENSE, "Normalised fidelity error "
                           "gradients:\n{}".format(self.fid_err_grad))

            if self.log_level <= logging.DEBUG:
                logger.debug("Gradient (sum sq norm): "
                             "{} ".format(self.grad_norm))

        return self.fid_err_grad

    def compute_fid_grad(self):
        """
        Calculates exact gradient of function wrt to each timeslot
        control amplitudes. Note these gradients are not normalised
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        n_ctrls = dyn.num_ctrls
        n_ts = dyn.num_tslots

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()
        for j in range(n_ctrls):
            for k in range(n_ts):
                fwd_evo = dyn._fwd_evo[k]   
                onto_evo = dyn._onto_evo[k+1]
                if dyn.oper_dtype == Qobj:
                    g = (onto_evo*dyn._get_prop_grad(k, j)*fwd_evo).tr()
                else:
                    g = _trace(onto_evo.dot(
                                dyn._get_prop_grad(k, j)).dot(fwd_evo))
                grad[k, j] = g
        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st
        return grad


class FidCompTraceDiff(FidelityComputer):
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

    Attributes
    ----------
    scale_factor : float
        The fidelity error calculated is of some arbitary scale. This
        factor can be used to scale the fidelity error such that it may
        represent some physical measure
        If None is given then it is caculated as 1/2N, where N
        is the dimension of the drift, when the Dynamics are initialised.
    """

    def reset(self):
        FidelityComputer.reset(self)
        self.id_text = 'TRACEDIFF'
        self.scale_factor = None
        self.uses_onwd_evo = True
        if not self.parent.prop_computer.grad_exact:
            raise errors.UsageError(
                "This FidelityComputer can only be"
                " used with an exact gradient PropagatorComputer.")
        self.apply_params()

    def init_comp(self):
        """
        initialises the computer based on the configuration of the Dynamics
        Calculates the scale_factor is not already set
        """
        if self.scale_factor is None:
            self.scale_factor = 1.0 / (2.0*self.parent.get_drift_dim())
            if self.log_level <= logging.DEBUG:
                logger.debug("Scale factor calculated as {}".format(
                    self.scale_factor))

    def get_fid_err(self):
        """
        Gets the absolute error in the fidelity
        """
        if not self.fidelity_current:
            dyn = self.parent
            dyn.compute_evolution()
            n_ts = dyn.num_tslots
            evo_final = dyn._fwd_evo[n_ts]
            evo_f_diff = dyn._target - evo_final
            if self.log_level <= logging.DEBUG_VERBOSE:
                logger.log(logging.DEBUG_VERBOSE, "Calculating TraceDiff "
                           "fidelity...\n Target:\n{}\n Evo final:\n{}\n"
                           "Evo final diff:\n{}".format(dyn._target, evo_final,
                                                        evo_f_diff))

            # Calculate the fidelity error using the trace difference norm
            # Note that the value should have not imagnary part, so using
            # np.real, just avoids the complex casting warning
            if dyn.oper_dtype == Qobj:
                self.fid_err = self.scale_factor*np.real(
                        (evo_f_diff.dag()*evo_f_diff).tr())
            else:
                self.fid_err = self.scale_factor*np.real(_trace(
                        evo_f_diff.conj().T.dot(evo_f_diff)))

            if np.isnan(self.fid_err):
                self.fid_err = np.Inf

            if dyn.stats is not None:
                    dyn.stats.num_fidelity_computes += 1

            self.fidelity_current = True
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity error: {}".format(self.fid_err))

        return self.fid_err

    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates
        (although this is not typically found to happen)
        """
        if not self.fid_err_grad_current:
            dyn = self.parent
            self.fid_err_grad = self.compute_fid_err_grad()
            self.fid_err_grad_current = True
            if dyn.stats is not None:
                dyn.stats.num_grad_computes += 1

            self.grad_norm = np.sqrt(np.sum(self.fid_err_grad**2))
            if self.log_level <= logging.DEBUG_INTENSE:
                logger.log(logging.DEBUG_INTENSE, "fidelity error gradients:\n"
                           "{}".format(self.fid_err_grad))

            if self.log_level <= logging.DEBUG:
                logger.debug("Gradient norm: "
                             "{} ".format(self.grad_norm))

        return self.fid_err_grad

    def compute_fid_err_grad(self):
        """
        Calculate exact gradient of the fidelity error function
        wrt to each timeslot control amplitudes.
        Uses the trace difference norm fidelity
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        n_ctrls = dyn.num_ctrls
        n_ts = dyn.num_tslots

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls])

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()


        evo_final = dyn._fwd_evo[n_ts]
        evo_f_diff = dyn._target - evo_final
        for j in range(n_ctrls):
            for k in range(n_ts):
                fwd_evo = dyn._fwd_evo[k]
                if dyn.oper_dtype == Qobj:
                    evo_grad = dyn._get_prop_grad(k, j)*fwd_evo
                    if k+1 < n_ts:
                        evo_grad = dyn._onwd_evo[k+1]*evo_grad
                    # Note that the value should have not imagnary part, so
                    # using np.real, just avoids the complex casting warning
                    g = -2*self.scale_factor*np.real(
                                    (evo_f_diff.dag()*evo_grad).tr())
                else:
                    evo_grad = dyn._get_prop_grad(k, j).dot(fwd_evo)
                    if k+1 < n_ts:
                        evo_grad = dyn._onwd_evo[k+1].dot(evo_grad)
                    g = -2*self.scale_factor*np.real(_trace(
                                    evo_f_diff.conj().T.dot(evo_grad)))
                if np.isnan(g):
                    g = np.Inf

                grad[k, j] = g
        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st
        return grad


class FidCompTraceDiffApprox(FidCompTraceDiff):
    """
    As FidCompTraceDiff, except uses the finite difference method to
    compute approximate gradients

    Attributes
    ----------
    epsilon : float
        control amplitude offset to use when approximating the gradient wrt
        a timeslot control amplitude
    
    """
    def reset(self):
        FidelityComputer.reset(self)
        self.id_text = 'TDAPPROX'
        self.uses_onwd_evo = True
        self.scale_factor = None
        self.epsilon = 0.001
        self.apply_params()

    def compute_fid_err_grad(self):
        """
        Calculates gradient of function wrt to each timeslot
        control amplitudes. Note these gradients are not normalised
        They are calulated
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        prop_comp = dyn.prop_computer
        n_ctrls = dyn.num_ctrls
        n_ts = dyn.num_tslots

        if self.log_level >= logging.DEBUG:
            logger.debug("Computing fidelity error gradient")
        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls])

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()
        curr_fid_err = self.get_fid_err()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()

        for j in range(n_ctrls):
            for k in range(n_ts):
                fwd_evo = dyn._fwd_evo[k]
                prop_eps = prop_comp._compute_diff_prop(k, j, self.epsilon)
                if dyn.oper_dtype == Qobj:
                    evo_final_eps = fwd_evo*prop_eps
                    if k+1 < n_ts:
                        evo_final_eps = evo_final_eps*dyn._onwd_evo[k+1]
                    evo_f_diff_eps = dyn._target - evo_final_eps
                    # Note that the value should have not imagnary part, so
                    # using np.real, just avoids the complex casting warning
                    fid_err_eps = self.scale_factor*np.real(
                        (evo_f_diff_eps.dag()*evo_f_diff_eps).tr())
                else:
                    evo_final_eps = fwd_evo.dot(prop_eps)
                    if k+1 < n_ts:
                        evo_final_eps = evo_final_eps.dot(dyn._onwd_evo[k+1])
                    evo_f_diff_eps = dyn._target - evo_final_eps
                    fid_err_eps = self.scale_factor*np.real(_trace(
                        evo_f_diff_eps.conj().T.dot(evo_f_diff_eps)))

                g = (fid_err_eps - curr_fid_err)/self.epsilon
                if np.isnan(g):
                    g = np.Inf

                grad[k, j] = g

        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st

        return grad

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
Classes that define the dynamics of the (quantum) system and target evolution
to be optimised.
The contols are also defined here, i.e. the dynamics generators (Hamiltonians,
Limbladians etc). The dynamics for the time slices are calculated here, along
with the evolution as determined by the control amplitudes.

See the subclass descriptions and choose the appropriate class for the
application. The choice depends on the type of matrix used to define
the dynamics.

These class implement functions for getting the dynamics generators for
the combined (drift + ctrls) dynamics with the approriate operator applied

Note the methods in these classes were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
"""
import os
import numpy as np
import scipy.linalg as la
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors
import qutip.control.tslotcomp as tslotcomp
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.control.symplectic as sympl

def _is_string(var):
    try:
        if isinstance(var, basestring):
            return True
    except NameError:
        try:
            if isinstance(var, str):
                return True
        except:
            return False
    except:
        return False
        
    return False

class Dynamics:
    """
    This is a base class only. See subclass descriptions and choose an
    appropriate one for the application.

    Note that initialize_controls must be called before any of the methods
    can be used.

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

    stats : Stats
        Attributes of which give performance stats for the optimisation
        set to None to reduce overhead of calculating stats.
        Note it is (usually) shared with the Optimizer object

    tslot_computer : TimeslotComputer (subclass instance)
        Used to manage when the timeslot dynamics
        generators, propagators, gradients etc are updated

    prop_computer : PropagatorComputer (subclass instance)
        Used to compute the propagators and their gradients

    fid_computer : FidelityComputer (subclass instance)
        Used to computer the fidelity error and the fidelity error
        gradient.

    num_tslots : integer
        Number of timeslots, aka timeslices

    num_ctrls : integer
        Number of controls.
        Note this is set when get_num_ctrls is called based on the
        length of ctrl_dyn_gen

    evo_time : float
        Total time for the evolution

    tau : array[num_tslots] of float
        Duration of each timeslot
        Note that if this is set before initialize_controls is called
        then num_tslots and evo_time are calculated from tau, otherwise
        tau is generated from num_tslots and evo_time, that is
        equal size time slices

    time : array[num_tslots+1] of float
        Cumulative time for the evolution, that is the time at the start
        of each time slice

    drift_dyn_gen : Qobj
        Drift or system dynamics generator
        Matrix defining the underlying dynamics of the system

    ctrl_dyn_gen : List of Qobj
        Control dynamics generator: ctrl_dyn_gen ()
        List of matrices defining the control dynamics

    initial : Qobj
        Starting state / gate
        The matrix giving the initial state / gate, i.e. at time 0
        Typically the identity

    target : Qobj
        Target state / gate:
        The matrix giving the desired state / gate for the evolution

    ctrl_amps : array[num_tslots, num_ctrls] of float
        Control amplitudes
        The amplitude (scale factor) for each control in each timeslot

    initial_ctrl_scaling : float
        Scale factor applied to be applied the control amplitudes
        when they are initialised
        This is used by the PulseGens rather than in any fucntions in
        this class

    self.initial_ctrl_offset = 0.0
        Linear offset applied to be applied the control amplitudes
        when they are initialised
        This is used by the PulseGens rather than in any fucntions in
        this class

    dyn_gen : List of Qobj
        Dynamics generators
        the combined drift and control dynamics generators
        for each timeslot

    prop : list of Qobj
        Propagators - used to calculate time evolution from one
        timeslot to the next

    prop_grad : array[num_tslots, num_ctrls] of Qobj
        Propagator gradient (exact gradients only)
        Array  of matrices that give the gradient
        with respect to the control amplitudes in a timeslot
        Note this attribute is only created when the selected
        PropagatorComputer is an exact gradient type.

    evo_init2t : List of Qobj
        Forward evolution (or propagation)
        the time evolution operator from the initial state / gate to the
        specified timeslot as generated by the dyn_gen

    evo_t2end : List of Qobj
        Onward evolution (or propagation)
        the time evolution operator from the specified timeslot to
        end of the evolution time as generated by the dyn_gen

    evo_t2targ : List of Qobj
        'Backward' List of Qobj propagation
        the overlap of the onward propagation with the inverse of the
        target.
        Note this is only used (so far) by the unitary dynamics fidelity

    evo_current : Boolean
        Used to flag that the dynamics used to calculate the evolution
        operators is current. It is set to False when the amplitudes
        change

    decomp_curr : List of boolean
        Indicates whether the diagonalisation for the timeslot is fresh,
        it is set to false when the dyn_gen for the timeslot is changed
        Only used when the PropagatorComputer uses diagonalisation

    dyn_gen_eigenvectors : List of array[drift_dyn_gen.shape]
        Eigenvectors of the dynamics generators
        Used for calculating the propagators and their gradients
        Only used when the PropagatorComputer uses diagonalisation

    prop_eigen : List of array[drift_dyn_gen.shape]
        Propagator in diagonalised basis of the combined dynamics generator
        Used for calculating the propagators and their gradients
        Only used when the PropagatorComputer uses diagonalisation

    dyn_gen_factormatrix : List of array[drift_dyn_gen.shape]
        Matrix of scaling factors calculated duing the decomposition
        Used for calculating the propagator gradients
        Only used when the PropagatorComputer uses diagonalisation

    fact_mat_round_prec : float
        Rounding precision used when calculating the factor matrix
        to determine if two eigenvalues are equivalent
        Only used when the PropagatorComputer uses diagonalisation

    def_amps_fname : string
        Default name for the output used when save_amps is called

    """
    def __init__(self, optimconfig, params=None):
        self.config = optimconfig
        self.params = params
        self.reset()

    def reset(self):
        # Link to optimiser object if self is linked to one
        self.parent = None
        # Main functional attributes
        self.evo_time = 1
        self.num_tslots = 10
        self.tau = None
        self.time = None
        self.initial = None
        self.target = None
        self.ctrl_amps = None
        self.initial_ctrl_scaling = 1.0
        self.initial_ctrl_offset = 0.0
        self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self.dyn_gen = None
        self.prop = None
        self.prop_grad = None
        self.evo_init2t = None
        self.evo_t2end = None
        self.evo_t2targ = None
        # Atrributes used in diagonalisation
        self.decomp_curr = None
        self.prop_eigen = None
        self.dyn_gen_eigenvectors = None
        self.dyn_gen_factormatrix = None
        self.fact_mat_round_prec = 1e-10

        # Debug and information attribs
        self.stats = None
        self.id_text = 'DYN_BASE'
        self.def_amps_fname = "ctrl_amps.txt"
        self.set_log_level(self.config.log_level)
        # Internal flags
        self._dyn_gen_mapped = False
        self._timeslots_initialized = False
        self._ctrls_initialized = False
        
        self.apply_params()

        # Create the computing objects
        self._create_computers()

        self.clear()
        
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
                
    def set_log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        self.log_level = lvl
        logger.setLevel(lvl)

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to UpdateAll
        # can be set to DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    def clear(self):
        self.ctrl_amps = None
        self.evo_current = False
        if self.fid_computer is not None:
            self.fid_computer.clear()

    def init_timeslots(self):
        """
        Generate the timeslot duration array 'tau' based on the evo_time
        and num_tslots attributes, unless the tau attribute is already set
        in which case this step in ignored
        Generate the cumulative time array 'time' based on the tau values
        """
        # set the time intervals to be equal timeslices of the total if
        # the have not been set already (as part of user config)
        if self.tau is None:
            self.tau = np.ones(self.num_tslots, dtype='f') * \
                self.evo_time/self.num_tslots
        else:
            self.num_tslots = len(self.tau)
            self.evo_time = np.sum(self.tau)

        self.time = np.zeros(self.num_tslots+1, dtype=float)
        # set the cumulative time by summing the time intervals
        for t in range(self.num_tslots):
            self.time[t+1] = self.time[t] + self.tau[t]
        
        self._timeslots_initialized = True
        
    def _init_lists(self):
        """
        Create the container lists / arrays for the:
        dynamics generations, propagators, and evolutions etc
        Set the time slices and cumulative time
        """

        # Create containers for control Hamiltonian etc
        shp = self.drift_dyn_gen.shape
        # set H to be just empty float arrays with the shape of H
        self.dyn_gen = [Qobj(shape=shp)
                        for x in range(self.num_tslots)]
        # the exponetiation of H. Just empty float arrays with the shape of H
        self.prop = [Qobj(shape=shp)
                     for x in range(self.num_tslots)]
        if self.prop_computer.grad_exact:
            self.prop_grad = np.empty([self.num_tslots, self.get_num_ctrls()],
                                      dtype=object)
        # Time evolution operator (forward propagation)
        self.evo_init2t = [Qobj(shape=shp)
                           for x in range(self.num_tslots + 1)]
        self.evo_init2t[0] = self.initial
        if self.fid_computer.uses_evo_t2end:
            # Time evolution operator (onward propagation)
            self.evo_t2end = [Qobj(shape=shp)
                              for x in range(self.num_tslots)]
        if self.fid_computer.uses_evo_t2targ:
            # Onward propagation overlap with inverse target
            self.evo_t2targ = [Qobj(shape=shp)
                               for x in range(self.num_tslots + 1)]
            self.evo_t2targ[-1] = self.get_owd_evo_target()

        if isinstance(self.prop_computer, propcomp.PropCompDiag):
            self._create_decomp_lists()

    def _create_decomp_lists(self):
        """
        Create lists that will hold the eigen decomposition
        used in calculating propagators and gradients
        Note: used with PropCompDiag propagator calcs
        """
        shp = self.drift_dyn_gen.shape
        n_ts = self.num_tslots
        self.decomp_curr = [False for x in range(n_ts)]
        self.prop_eigen = \
            [np.empty(shp[0], dtype=complex) for x in range(n_ts)]
        self.dyn_gen_eigenvectors = \
            [np.empty(shp, dtype=complex) for x in range(n_ts)]
        self.dyn_gen_factormatrix = \
            [np.empty(shp, dtype=complex) for x in range(n_ts)]

    def _check_test_out_files(self):
        cfg = self.config
        if cfg.any_test_files():
            if cfg.check_create_test_out_dir():
                if self.stats is None:
                    logger.warn("Cannot output test files when stats"
                                " attribute is not set.")
                    cfg.clear_test_out_flags()

    def initialize_controls(self, amps, init_tslots=True):
        """
        Set the initial control amplitudes and time slices
        Note this must be called after the configuration is complete
        before any dynamics can be calculated
        """
        self._check_test_out_files()

        if not isinstance(self.prop_computer, propcomp.PropagatorComputer):
            raise errors.UsageError(
                "No prop_computer (propagator computer) "
                "set. A default should be assigned by the Dynamics subclass")

        if not isinstance(self.tslot_computer, tslotcomp.TimeslotComputer):
            raise errors.UsageError(
                "No tslot_computer (Timeslot computer)"
                " set. A default should be assigned by the Dynamics class")

        if not isinstance(self.fid_computer, fidcomp.FidelityComputer):
            raise errors.UsageError(
                "No fid_computer (Fidelity computer)"
                " set. A default should be assigned by the Dynamics subclass")

        self.ctrl_amps = None
        # Note this call is made just to initialise the num_ctrls attrib
        self.get_num_ctrls()

        if not self._timeslots_initialized:
            init_tslots = True
        if init_tslots:
            self.init_timeslots()
        self._init_lists()
        self.tslot_computer.init_comp()
        self.fid_computer.init_comp()
        self._ctrls_initialized = True
        self.update_ctrl_amps(amps)

    def check_ctrls_initialized(self):
        if not self._ctrls_initialized:
            raise errors.UsageError(
                "Controls not initialised. "
                "Ensure Dynamics.initialize_controls has been "
                "executed with the initial control amplitudes.")

    def get_amp_times(self):
        return self.time[:self.num_tslots]
        
    def save_amps(self, file_name=None, times=None, amps=None, verbose=False):
        """
        Save a file with the current control amplitudes in each timeslot
        The first column in the file will be the start time of the slot

        Parameters
        ----------
        file_name : string
            Name of the file
            If None given the def_amps_fname attribuite will be used

        times : List type (or string)
            List / array of the start times for each slot
            If None given this will be retrieved through get_amp_times()
            If 'exclude' then times will not be saved in the file, just
            the amplitudes

        amps : Array[num_tslots, num_ctrls]
            Amplitudes to be saved
            If None given the ctrl_amps attribute will be used

        verbose : Boolean
            If True then an info message will be logged
        """
        self.check_ctrls_initialized()

        inctimes = True
        if file_name is None:
            file_name = self.def_amps_fname
        if amps is None:
            amps = self.ctrl_amps
        if times is None:
            times = self.get_amp_times()
        else:
            if _is_string(times):
                if times.lower() == 'exclude':
                    inctimes = False
                else:
                    logger.warn("Unknown option for times '{}' "
                                "when saving amplitudes".format(times))
                    times = self.get_amp_times()
            
        try:
            if inctimes:
                shp = amps.shape
                data = np.empty([shp[0], shp[1] + 1], dtype=float)
                data[:, 0] = times
                data[:, 1:] = amps
            else:
                data = amps

            np.savetxt(file_name, data, delimiter='\t', fmt='%14.6g')

            if verbose:
                logger.info("Amplitudes saved to file: " + file_name)
        except Exception as e:
            logger.error("Failed to save amplitudes due to underling "
                         "error: {}".format(e))

    def update_ctrl_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        The actual work is completed by the compare_amps method of the
        timeslot computer
        """

        if self.log_level <= logging.DEBUG_INTENSE:
            logger.log(logging.DEBUG_INTENSE, "Updating amplitudes...\n"
                       "Current control amplitudes:\n" + str(self.ctrl_amps) +
                       "\n(potenially) new amplitudes:\n" + str(new_amps))

        if not self.tslot_computer.compare_amps(new_amps):
            if self.config.test_out_amps:
                fname = "amps_{}_{}_{}_call{}{}".format(
                    self.id_text,
                    self.prop_computer.id_text,
                    self.fid_computer.id_text,
                    self.stats.num_ctrl_amp_updates,
                    self.config.test_out_f_ext)

                fpath = os.path.join(self.config.test_out_dir, fname)
                self.save_amps(fpath, verbose=True)

    def flag_system_changed(self):
        """
        Flag eveolution, fidelity and gradients as needing recalculation
        """
        self.evo_current = False
        self.fid_computer.flag_system_changed()

    def get_drift_dim(self):
        """
        Returns the size of the matrix that defines the drift dynamics
        that is assuming the drift is NxN, then this returns N
        """
        if not isinstance(self.drift_dyn_gen, np.ndarray):
            raise TypeError("Cannot get drift dimension, "
                            "as drift not set (correctly).")
        return self.drift_dyn_gen.shape[0]

    def get_num_ctrls(self):
        """
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively
        subsequently
        """
        self.num_ctrls = len(self.ctrl_dyn_gen)
        return self.num_ctrls

    def get_owd_evo_target(self):
        """
        Get the inverse of the target.
        Used for calculating the 'backward' evolution
        """
        return la.inv(self.target)

    def combine_dyn_gen(self, k):
        """
        Computes the dynamics generator for a given timeslot
        The is the combined Hamiltion for unitary systems
        """
        dg = self.drift_dyn_gen
        for j in range(self.get_num_ctrls()):
            dg = dg + self.ctrl_amps[k, j]*self.ctrl_dyn_gen[j]
        return dg

    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        Not implemented in the base class. Choose a subclass
        """
        raise errors.UsageError("Not implemented in the baseclass."
                                " Choose a subclass")

    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        Not implemented in the base class. Choose a subclass
        """
        raise errors.UsageError("Not implemented in the baseclass."
                                " Choose a subclass")

    def compute_evolution(self):
        """
        Recalculate the time evolution operators
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary
        Actual work is completed by the recompute_evolution method
        of the timeslot computer
        """

        # Check if values are already current, otherwise calculate all values
        if not self.evo_current:
            if self.log_level <= logging.DEBUG_VERBOSE:
                logger.log(logging.DEBUG_VERBOSE, "Computing evolution")
            self.tslot_computer.recompute_evolution()
            self.evo_current = True
            return True
        else:
            return False

    def ensure_decomp_curr(self, k):
        """
        Checks to see if the diagonalisation has been completed since
        the last update of the dynamics generators
        (after the amplitude update)
        If not then the diagonlisation is completed
        """
        if self.decomp_curr is None:
            raise errors.UsageError("Decomp lists have not been created")
        if not self.decomp_curr[k]:
            self.spectral_decomp(k)

    def spectral_decomp(self, k):
        """
        Calculate the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient
        Not implemented in this base class, because the method is specific
        to the matrix type
        """
        raise errors.UsageError("Decomposition cannot be completed by "
                                "this class. Try a(nother) subclass")


class DynamicsGenMat(Dynamics):
    """
    This sub class can be used for any system where no additional
    operator is applied to the dynamics generator before calculating
    the propagator, e.g. classical dynamics, Lindbladian
    """
    def reset(self):
        Dynamics.reset(self)
        self.id_text = 'GEN_MAT'

    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        This base class method simply returns dyn_gen[k]
        other subclass methods will include some factor
        """
        return self.dyn_gen[k]

    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        This base class method simply returns ctrl_dyn_gen[j]
        other subclass methods will include some factor
        """
        return self.ctrl_dyn_gen[j]


class DynamicsUnitary(Dynamics):
    """
    This is the subclass to use for systems with dynamics described by
    unitary matrices. E.g. closed systems with Hermitian Hamiltonians
    Note a matrix diagonalisation is used to compute the exponent
    The eigen decomposition is also used to calculate the propagator gradient.
    The method is taken from DYNAMO (see file header)

    Attributes
    ----------
    drift_ham : Qobj
        This is the drift Hamiltonian for unitary dynamics
        It is mapped to drift_dyn_gen during initialize_controls

    ctrl_ham : List of Qobj
        These are the control Hamiltonians for unitary dynamics
        It is mapped to ctrl_dyn_gen during initialize_controls

    H : List of Qobj
        The combined drift and control Hamiltonians for each timeslot
        These are the dynamics generators for unitary dynamics.
        It is mapped to dyn_gen during initialize_controls
    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = 'UNIT'
        self.drift_ham = None
        self.ctrl_ham = None
        self.H = None
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        # set the default fidelity computer
        self.fid_computer = fidcomp.FidCompUnitary(self)
        # set the default propagator computer
        self.prop_computer = propcomp.PropCompDiag(self)

    def initialize_controls(self, amplitudes, init_tslots=True):
        # Either the _dyn_gen or _ham names can be used
        # This assumes that one or other has been set in the configuration

        self._map_dyn_gen_to_ham()
        Dynamics.initialize_controls(self, amplitudes, init_tslots=init_tslots)
        self.H = self.dyn_gen

    def _map_dyn_gen_to_ham(self):
        if self.drift_dyn_gen is None:
            self.drift_dyn_gen = self.drift_ham
        else:
            self.drift_ham = self.drift_dyn_gen

        if self.ctrl_dyn_gen is None:
            self.ctrl_dyn_gen = self.ctrl_ham
        else:
            self.ctrl_ham = self.ctrl_dyn_gen

        self._dyn_gen_mapped = True

    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        including the -i factor
        """
        return -1j*self.dyn_gen[k]

    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        including the -i factor
        """
        return -1j*self.ctrl_dyn_gen[j]

    def get_num_ctrls(self):
        if not self._dyn_gen_mapped:
            self._map_dyn_gen_to_ham()
        return Dynamics.get_num_ctrls(self)

    def get_owd_evo_target(self):
        return self.target.dag()

    def spectral_decomp(self, k):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient
        """
        H = self.H[k].full()
        # assuming H is an nxn matrix, find n
        n = H.shape[0]
        # returns row vector of eigen values,
        # columns with the eigenvectors
        eig_val, eig_vec = np.linalg.eig(H)

        # Calculate the propagator in the diagonalised basis
        eig_val_tau = -1j*eig_val*self.tau[k]
        prop_eig = np.exp(eig_val_tau)

        # Generate the factor matrix through the differences
        # between each of the eigenvectors and the exponentiations
        # create nxn matrix where each eigen val is repeated n times
        # down the columns
        o = np.ones([n, n])
        eig_val_cols = eig_val_tau*o
        # calculate all the differences by subtracting it from its transpose
        eig_val_diffs = eig_val_cols - eig_val_cols.T
        # repeat for the propagator
        prop_eig_cols = prop_eig*o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T
        # the factor matrix is the elementwise quotient of the
        # differeneces between the exponentiated eigen vals and the
        # differences between the eigen vals
        # need to avoid division by zero that would arise due to denegerate
        # eigenvalues and the diagonals
        degen_mask = np.abs(eig_val_diffs) < self.fact_mat_round_prec
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        # for degenerate eigenvalues the factor is just the exponent
        factors[degen_mask] = prop_eig_cols[degen_mask]

        # Store eigenvals and eigenvectors for use by other functions, e.g.
        # gradient_exact
        self.decomp_curr[k] = True
        self.prop_eigen[k] = prop_eig
        self.dyn_gen_eigenvectors[k] = eig_vec
        self.dyn_gen_factormatrix[k] = factors


class DynamicsSymplectic(Dynamics):
    """
    Symplectic systems
    This is the subclass to use for systems where the dynamics is described
    by symplectic matrices, e.g. coupled oscillators, quantum optics

    Attributes
    ----------
    omega : array[drift_dyn_gen.shape]
        matrix used in the calculation of propagators (time evolution)
        with symplectic systems.
    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = 'SYMPL'
        self.omega = None
        self.grad_exact = True
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    def get_omega(self):
        if self.omega is None:
            n = self.drift_dyn_gen.shape[0] // 2
            self.omega = Qobj(sympl.calc_omega(n))

        return self.omega

    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        multiplied by omega
        """
        o = self.get_omega()
        return -self.dyn_gen[k].dot(o)

    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        multiplied by omega
        """
        o = self.get_omega()
        return -self.ctrl_dyn_gen[j].dot(o)

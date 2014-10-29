# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:40:45 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

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
import errors as errors
import tslotcomp
import fidcomp
import propcomp
import symplectic as sympl

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class Dynamics:
    """
    This is a base class only. See subclass descriptions and choose an
    appropriate one for the application.
    Container for:
        Drift or system dynamics generator: drift_dyn_gen (drift_ham)
            Matrix defining the underlying dynamics of the system
            (This is the drift Hamiltonian for unitary dynamics)
            
        Control dynamics generator: Ctrl_dyn_gen (Ctrl_ham)
            List of matrices defining the control dynamics
            (These is the control Hamiltonians for unitary dynamics)
        
        Starting state / gate: initial
            The matrix giving the initial state / gate, i.e. at time 0
            Typically the identity
            
        Target state / gate: target
            The matrix giving the desired state / gate for the evolution        
        
        Control amplitudes:
            The amplitude (scale factor) for each control in each timeslot
            
        Dynamics generators: Dyn_gen (H)
            the combined drift and control generators for each timeslot
            
        Propagators: Prop
            List of propagators 
            used to calculate time evolution from one timeslot to the next
            
        Propagator gradient (exact gradients only): Prop_grad
            Array(N timeslots x N ctrls) of matrices that give the gradient
            with respect to the ctrl amplitudes in a timeslot
                    
        Forward propagation: Evo_init2t
            the time evolution operator from the initial to the 
            specified timeslot
                    
        Onward propagation: Evo_t2end
            the time evolution operator from the specified timeslot to
            end of the evolution time
            
        'Backward' propagation: Evo_t2targ
            the overlap of the onward propagation with the inverse of the 
            target
            
        NOTE: that the Dyn_gen attributes are mapped to H/ham attributes
            for systems with unitary dynamics (see _map_dyn_gen_to_ham)
    
    Note that initialize_controls must be called before any of the methods
    can be used.
    """ 
    def __init__(self, optimconfig):
        self.config = optimconfig
        self.reset()
        
    def reset(self):
        self.msg_level = self.config.msg_level
        self.test_out_files = self.config.test_out_files
        # Total time for the evolution
        self.evo_time = 0
        # Number of time slots (slices)
        self.num_tslots = 0
        # Duration of each timeslot
        self.tau = None
        # Cumulative time
        self.time = None
        # Initial state / gate
        self.initial = None
        # Target state / gate
        self.target = None
        # Control amplitudes
        self.ctrl_amps = None
        # Drift or system dynamics generator, e.g. Hamiltonian, Limbladian
        self.drift_dyn_gen = None
        # List of dynamics generators for the controls
        self.Ctrl_dyn_gen = None
        # Dyn_gen are the dynamics generators, e.g. Hamiltonian, Limbladian 
        self.Dyn_gen = None
        # Progators from time slot k to k+1
        self.Prop = None
        # Gradient of propagator wrt the control amplitude in the timeslot
        self.Prop_grad = None
        # Evolution from initial (k=0) to given time slice
        self.Evo_init2t = None
        # Evolution from given time slice to end of evolution time
        self.Evo_t2end = None
        # Evolution from given time slice overlapped with inverse target
        self.Evo_t2targ = None
        # List to indicate whether the diagonisation for the timeslot is fresh
        self.Decomp_curr = None
        # List of propagators in the diagonalised basis
        self.Prop_eigen = None
        # Eigenvectors of the dynamics generators
        self.Dyn_gen_eigenvectors = None
        # Factor matrix used to calculate the propagotor gradient
        # calculated duing the decomposition
        self.Dyn_gen_factormatrix = None
        # Rounding precision used when calculating the factor matrix
        self.fact_mat_round_prec = 1e-10
        # Initial amplitude scaling and offset (from 0)
        self.initial_ctrl_scaling = 1.0
        self.initial_ctrl_offset = 0.0
        
        self._create_computers()
        
        self.def_amps_fname = "ctrl_amps.txt"
    
        # This is the object used to collect stats for the optimisation
        # If it is not set, then stats are not collected, other than
        # those defined as properties of this object
        # Note it is (usually) shared with the optimiser object
        self.stats = None
        
        self._dyn_gen_mapped = False
        
        self.clear()
        
    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if (self.config.amp_update_mode == 'DYNAMIC'):
            self.tslot_computer = tslotcomp.TSlotComp_DynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotComp_UpdateAll(self)
        
        self.prop_computer = propcomp.PropComp_Frechet(self)
        self.fid_computer = fidcomp.FidComp_TraceDiff(self)
        
        
    def clear(self):
        self.evo_current = False
        if (self.fid_computer != None):
            self.fid_computer.clear()
        
    def _init_lists(self):
        """
        Create the container lists / arrays for the:
        dynamics generations, propagators, and evolutions etc
        Set the time slices and cumulative time
        """

        # set the time intervals to be equal timeslices of the total if 
        # the have not been set already (as part of user config)
        if (self.tau == None):
            self.tau =  np.ones(self.num_tslots, dtype = 'f') * \
                            self.evo_time/self.num_tslots
        self.time = np.zeros(self.num_tslots+1, dtype=float)
        # set the cumulative time by summing the time intervals
        for t in range(self.num_tslots):
            self.time[t+1] = self.time[t] + self.tau[t]
        # Create containers for control Hamiltonian etc
        shp = self.drift_dyn_gen.shape
        # set H to be just empty float arrays with the shape of H
        self.Dyn_gen = [np.empty(shp, dtype=complex) \
                        for x in xrange(self.num_tslots)]
        # the exponetiation of H. Just empty float arrays with the shape of H
        self.Prop = [np.empty(shp, dtype=complex) \
                        for x in xrange(self.num_tslots)]
        if (self.prop_computer.grad_exact):
            self.Prop_grad = np.empty([self.num_tslots, self.get_num_ctrls()],\
                        dtype = np.ndarray)   
        # Time evolution operator (forward propagation)
        self.Evo_init2t = [np.empty(shp, dtype=complex) \
                        for x in xrange(self.num_tslots + 1)]
        self.Evo_init2t[0] = self.initial
        if (self.fid_computer.uses_evo_t2end):
            # Time evolution operator (onward propagation)
            self.Evo_t2end = [np.empty(shp, dtype=complex) \
                            for x in xrange(self.num_tslots)]
        if (self.fid_computer.uses_evo_t2targ):
            # Onward propagation overlap with inverse target
            self.Evo_t2targ = [np.empty(shp, dtype=complex) \
                            for x in xrange(self.num_tslots + 1)]
            self.Evo_t2targ[-1] = self.get_owd_evo_target()

        if (isinstance(self.prop_computer, propcomp.PropComp_Diag)):
            self._create_decomp_lists()
            
    def _create_decomp_lists(self):
        """
        Create lists that will hold the eigen decomposition
        used in calculating propagators and gradients
        Note used with PropComp_Diag propagator calcs
        """
        shp = self.drift_dyn_gen.shape
        nTS = self.num_tslots
        self.Decomp_curr = \
            [False for x in xrange(nTS)]
        # Propagators in diagonalised basis
        self.Prop_eigen = \
            [np.empty(shp[0], dtype=complex) for x in xrange(nTS)]
        self.Dyn_gen_eigenvectors = \
            [np.empty(shp, dtype=complex) for x in xrange(nTS)]
        self.Dyn_gen_factormatrix = \
            [np.empty(shp, dtype=complex) for x in xrange(nTS)]
             
    def initialize_controls(self, amps):
        """
        Set the initial control amplitudes and time slices
        Note this must be called after the configuration is complete
        before any dynamics can be calculated
        """
        
        if (self.test_out_files >= 1 and self.stats == None):
            f = self.__class__.__name__ + ".initialize_controls"
            m = "Cannot output test files when stats object is not set"
            raise errors.UsageError(funcname=f, msg=m)
            
        if (not isinstance(self.prop_computer, propcomp.PropagatorComputer)):
            f = self.__class__.__name__ + ".initialize_controls"
            m = "No prop_computer (propagator computer) set." + \
                " A default should be assigned by the Dynamics subclass"
            raise errors.UsageError(funcname=f, msg=m)
            
        if (not isinstance(self.tslot_computer, tslotcomp.TimeslotComputer)):
            f = self.__class__.__name__ + ".initialize_controls"
            m = "No tslot_computer (Timeslot computer) set." + \
                " A default should be assigned by the Dynamics class"
            raise errors.UsageError(funcname=f, msg=m)
                
        if (not isinstance(self.fid_computer, fidcomp.FideliyComputer)):
            f = self.__class__.__name__ + ".initialize_controls"
            m = "No fid_computer (Fidelity computer) set." + \
                " A default should be assigned by the Dynamics subclass"
            raise errors.UsageError(funcname=f, msg=m)
        
        # Note this call is made just to initialise the num_ctrls attrib
        n_ctrls = self.get_num_ctrls()
        
        self._init_lists()
        self.tslot_computer.init_comp()
        self.fid_computer.init_comp()
        self.update_ctrl_amps(amps)
        
    def get_amp_times(self):
        return self.time[:self.num_tslots]
    
    def save_amps(self, file_name=None, times=None, amps=None):
        """
        Save a file with the current control amplitudes in each timeslot
        The first column in the file will be the start time of the slot
        """
        if (file_name == None):
            file_name = self.def_amps_fname
        if (amps == None):
            amps = self.ctrl_amps
        if (times == None):
            times = self.get_amp_times()
            
        shp = amps.shape
        data = np.empty([shp[0], shp[1] + 1], dtype=float)
        data[:, 0] = times
        data[:, 1:] = amps
        np.savetxt(file_name, data, delimiter='\t')
        if (self.msg_level >= 2):
            print "Amplitudes saved to file: " + file_name
            
    def update_ctrl_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        The actual work is completed by the compare_amps method of the
        timeslot computer
        """
        
        if (not self.tslot_computer.compare_amps(new_amps)):
            if (self.msg_level >= 3):
                print "self.ctrl_amps"
                print self.ctrl_amps
                print "New_amps"
                print new_amps
                
            if (self.test_out_files >= 1):
                fname = os.path.join("test_out", \
                        "amps_{}_{}_call{}.txt".format(self.config.dyn_type, \
                            self.config.fid_type, \
                            self.stats.num_ctrl_amp_updates))
                self.save_amps(fname)
        
    def flag_system_changed(self):
        """
        Flag eveolution, fidelity and gradients as needing recalculation
        """
        self.evo_current = False
        self.fid_computer.flag_system_changed()

    def get_num_ctrls(self):
        """
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively 
        subsequently
        """
        self.num_ctrls = len(self.Ctrl_dyn_gen)
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
        dg = np.asarray(self.drift_dyn_gen)
        for j in range(self.get_num_ctrls()):
            dg = dg + self.ctrl_amps[k, j]*np.asarray(self.Ctrl_dyn_gen[j])
        return dg
        
    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        Not implemented in the base class. Choose a subclass
        subclass methods will include some factor
        """
        return self.Dyn_gen[k]
        
    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        Not implemented in the base class. Choose a subclass
        """
        f = self.__class__.__name__ + ".get_ctrl_dyn_gen"
        m = "Not implemented in the baseclass. Choose a subclass"
        raise errors.UsageError(funcname=f, msg=m)
            
    def compute_evolution(self):
        """
        Recalculate the time evolution operators
        Dynamics generators (e.g. Hamiltonian) and 
        Prop (propagators) are calculated as necessary
        Actual work is completed by the recompute_evolution method
        of the timeslot computer
        """

        # Check if values are already current, otherwise calculate all values
        if (not self.evo_current):
            if (self.msg_level >= 2):
                print "Computing evolution"
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
        if (self.Decomp_curr is None):
            f = self.__class__.__name__ + ".ensure_decomp_curr"
            m = "Decomp lists have not been created"
            raise errors.UsageError(funcname=f, msg=m)
        if (not self.Decomp_curr[k]):
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
        f = self.__class__.__name__ + ".spectral_decomp"
        m = "Decomposition cannot be completed by this class. Try a subclass"
        raise errors.UsageError(funcname=f, msg=m)
            


# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

class Dynamics_GenMat(Dynamics):
    """
    This sub class can be used for any system where no additional
    operator is applied to the dynamics generator before calculating
    the propagator, e.g. classical dynamics, Limbladian
    """
    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        This base class method simply returns Dyn_gen[k]
        other subclass methods will include some factor
        """
        return self.Dyn_gen[k]
        
    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        This base class method simply returns Ctrl_dyn_gen[j]
        other subclass methods will include some factor
        """
        return self.Ctrl_dyn_gen[j]
        
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class Dynamics_Unitary(Dynamics):
    """
    This is the subclass to use for systems with dynamics described by 
    unitary matrices. E.g. closed systems with Hermitian Hamiltonians
    Note a matrix diagonalisation is used to compute the exponent
    The eigen decomposition is also used to calculate the propagator gradient.
    The method is taken from DYNAMO (see file header)
    """
            
    def reset(self):
        Dynamics.reset(self)
        self.drift_ham = None
        self.Ctrl_ham = None
        self.H = None

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if (self.config.amp_update_mode == 'DYNAMIC'):
            self.tslot_computer = tslotcomp.TSlotComp_DynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotComp_UpdateAll(self)
        
        # set the default fidelity computer
        self.fid_computer = fidcomp.FidComp_Unitary(self)
        # set the default propagator computer
        self.prop_computer = propcomp.PropComp_Diag(self)
        
    def initialize_controls(self, amplitudes):
        # Either the _dyn_gen or _ham names can be used
        # This assumes that one or other has been set in the configuration

        self._map_dyn_gen_to_ham()
        Dynamics.initialize_controls(self, amplitudes)
        self.H = self.Dyn_gen

    def _map_dyn_gen_to_ham(self):
        if (self.drift_dyn_gen == None):
            self.drift_dyn_gen = self.drift_ham
        else:
            self.drift_ham = self.drift_dyn_gen
        if (self.Ctrl_dyn_gen == None):
            self.Ctrl_dyn_gen = self.Ctrl_ham
        else:
            self.Ctrl_ham = self.Ctrl_dyn_gen
        
        self._dyn_gen_mapped = True
        
    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        including the -i factor
        """
        return -1j*self.Dyn_gen[k]
        
    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        including the -i factor
        """
        return -1j*self.Ctrl_dyn_gen[j]
        
    def get_num_ctrls(self):
        if (not self._dyn_gen_mapped):
            self._map_dyn_gen_to_ham()
        return Dynamics.get_num_ctrls(self)

    def get_owd_evo_target(self):
        return self.target.conj().T
    
    def spectral_decomp(self, k):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator 
        gradient
        """
        H = self.H[k]
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
        o = np.ones([n,n])
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
        self.Decomp_curr[k] = True
        self.Prop_eigen[k] = prop_eig
        self.Dyn_gen_eigenvectors[k] = eig_vec
        self.Dyn_gen_factormatrix[k] = factors
              
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
            
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class Dynamics_Sympl(Dynamics):
    """
    Symplectic systems
    This is the subclass to use for systems where the dynamics is described
    by symplectic matrices, e.g. coupled oscillators, quantum optics
    """
        
    def reset(self):
        Dynamics.reset(self)
        self.omega = None
        self.grad_exact = True

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if (self.config.amp_update_mode == 'DYNAMIC'):
            self.tslot_computer = tslotcomp.TSlotComp_DynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotComp_UpdateAll(self)
        
        self.prop_computer = propcomp.PropComp_Frechet(self)
        self.fid_computer = fidcomp.FidComp_TraceDiff(self)
        
    def get_omega(self):
        if (self.omega == None):
            n = self.drift_dyn_gen.shape[0]/2
            self.omega = sympl.calc_omega(n)
        
        return self.omega
        
    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        multiplied by omega
        """
        o = self.get_omega()
        return self.Dyn_gen[k].dot(o)
        
    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        multiplied by omega
        """
        o = self.get_omega()
        return self.Ctrl_dyn_gen[j].dot(o)
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]



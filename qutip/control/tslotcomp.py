# -*- coding: utf-8 -*-
"""
Created on Mon Jun 09 21:42:15 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Timeslot Computer
These classes determine which dynamics generators, propagators and evolutions
are recalculated when there is a control amplitude update. 
The timeslot computer processes the lists held by the dynamics object

The default (_UpdateAll) updates all of these each amp update, on the 
assumption that all amplitudes are changed each iteration. This is typical
when using optimisation methods like BFGS in the GRAPE algorithm

The alternative (_DynUpdate) assumes that only a subset of amplitudes 
are updated each iteration and attempts to minimise the number of expensive
calculations accordingly. This would be the appropriate class for Krotov type
methods. Note that the Stats_DynTsUpdate class must be used for stats
in conjunction with this class.
NOTE: AJGP 2011-10-2014: This _DynUpdate class currently has some bug,
no pressing need to fix it presently

If all amplitudes change at each update, then the behavior of the classes is 
equivalent. _UpdateAll is easier to understand and potentially slightly faster
in this situation.

Note the methods in the _DynUpdate class were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
"""

import os
import numpy as np
import timeit
import errors as errors
import utility as util

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class TimeslotComputer:
    """
    Base class for all Timeslot Computers
    Note: this must be instantiated with a Dynamics object, that is the
    container for the data that the methods operate on
    """
    def __init__(self, dynamics):
        self.parent = dynamics
        self.reset()
        
    def reset(self):
        self.msg_level = self.parent.msg_level
        
    def flag_all_calc_now(self):
        pass
        
    def init_comp(self):
        pass
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class TSlotComp_UpdateAll(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    
    def compare_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        Returns: True if amplitudes are the same, False if they have changed
        """
        changed = False
        dyn = self.parent
        if (dyn.ctrl_amps == None):
            # Flag fidelity and gradients as needing recalculation
            changed = True
            if (dyn.stats != None):
                dyn.stats.num_ctrl_amp_updates = 1
                dyn.stats.num_ctrl_amp_changes = len(new_amps.flat)
                dyn.stats.num_timeslot_changes = new_amps.shape[0]
        else:
            # create boolean array with same shape as ctrl_amps
            # True where value in New_amps differs, otherwise false
            changedAmps = dyn.ctrl_amps != new_amps
            if (np.any(changedAmps)):
                # Flag fidelity and gradients as needing recalculation
                changed = True
                if (self.msg_level >= 2):
                    print "{} amplitudes changed".format(changedAmps.sum())
                # *** update stats ***
                if (dyn.stats != None):
                    dyn.stats.num_ctrl_amp_updates += 1
                    dyn.stats.num_ctrl_amp_changes += changedAmps.sum()
                    changedTSmask = np.any(changedAmps, 1)
                    dyn.stats.num_timeslot_changes += changedTSmask.sum()
    
        if (changed):
            dyn.ctrl_amps = new_amps
            dyn.flag_system_changed()
            return False
        else:
            return True
            
    def recompute_evolution(self):                    
        """
        Recalculates the evolution operators.
        Dynamics generators (e.g. Hamiltonian) and 
        Prop (propagators) are calculated as necessary 
        """
        
        if (self.msg_level >= 2):
            print "recomputing evolution (UpdateAll)"

        dyn = self.parent
        prop_comp = dyn.prop_computer
        nTS = dyn.num_tslots
        nCtrls = dyn.get_num_ctrls()
        # calculate the Hamiltonians
        timeStart = timeit.default_timer()
        for k in range(nTS):
            dyn.Dyn_gen[k] = dyn.combine_dyn_gen(k)
            if (dyn.Decomp_curr is not None):
                dyn.Decomp_curr[k] = False
        if (dyn.stats != None):
            dyn.stats.wall_time_dyn_gen_compute += \
                                timeit.default_timer() - timeStart
        
        # calculate the propagators and the propagotor gradients
        timeStart = timeit.default_timer()
        for k in range(nTS):
            if (prop_comp.grad_exact):
                for j in range(nCtrls):
                    if (j == 0):
                        prop, propGrad = prop_comp.compute_prop_grad(k, j)
                        dyn.Prop[k] = prop
                        dyn.Prop_grad[k, j] = propGrad
                        if (self.msg_level >= 5):
                            print "propagator {}:".format(k)
                            print prop
                        if (dyn.test_out_files >= 3):
                            fname = os.path.join("test_out", \
                                    "propGrad_{}_j{}_k{}.txt".\
                                    format(dyn.config.dyn_type, j, k))
                            util.write_array_to_file(self.Prop_grad[k, j], \
                                    fname, dtype=complex)
                    else:
                        propGrad = prop_comp.compute_prop_grad(k, j, \
                                                    compute_prop=False)
                        dyn.Prop_grad[k, j] = propGrad
            else:
                dyn.Prop[k] = prop_comp.compute_propagator(k)
                
        if (dyn.stats != None):
            dyn.stats.wall_time_prop_compute += \
                                timeit.default_timer() - timeStart
    
        # compute the forward propagation
        timeStart = timeit.default_timer()
        R = range(1, nTS + 1)
        for k in R:
            dyn.Evo_init2t[k] = dyn.Prop[k-1].dot(dyn.Evo_init2t[k-1])

        if (dyn.stats != None):
            dyn.stats.wall_time_fwd_prop_compute += \
                                timeit.default_timer() - timeStart
            
        timeStart = timeit.default_timer()  
        # compute the onward propagation
        if (dyn.fid_computer.uses_evo_t2end):
            dyn.Evo_t2end[nTS - 1] = dyn.Prop[nTS - 1]
            R = range(nTS-2, -1, -1)
            for k in R:
                dyn.Evo_t2end[k] = dyn.Evo_t2end[k+1].dot(dyn.Prop[k])
                
        if (dyn.fid_computer.uses_evo_t2targ):
            R = range(nTS-1, -1, -1)
            for k in R:
                dyn.Evo_t2targ[k] = dyn.Evo_t2targ[k+1].dot(dyn.Prop[k])
            if (dyn.stats != None):
                dyn.stats.wall_time_onwd_prop_compute += \
                                    timeit.default_timer() - timeStart
                                    
    def get_timeslot_for_fidelity_calc(self):
        """
        Returns the timeslot index that will be used calculate current fidelity
        value. 
        This (default) method simply returns the last timeslot
        """
        return self.parent.num_tslots - 1
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class TSlotComp_DynUpdate(TimeslotComputer):
    """
    Timeslot Computer - Dynamic Update
    ********************************
    ***** CURRENTLY HAS ISSUES *****
    ***** AJGP 2014-10-02
    ***** and is therefore not being maintained
    ***** i.e. changes made to _UpdateAll are not being implemented here
    ********************************
    Updates only the dynamics generators, propagators and evolutions as 
    required when a subset of the ctrl amplitudes are updated.
    Will update all if all amps have changed.
    """
        
    def reset(self):
        self.Dyn_gen_recalc = None
        self.Prop_recalc = None
        self.Evo_init2t_recalc = None
        self.Evo_t2targ_recalc = None
        self.Dyn_gen_calc_now = None
        self.Prop_calc_now = None
        self.Evo_init2t_calc_now = None
        self.Evo_t2targ_calc_now = None
        TimeslotComputer.reset(self)
        
    def init_comp(self):
        """
        Initialise the flags
        """
        ####
        # These maps are used to determine what needs to be updated
        ####
        # Note _recalc means the value needs updating at some point
        # e.g. here no values have been set, except the initial and final
        # evolution operator vals (which never change) and hence all other
        # values are set as requiring calculation.
        nTS = self.parent.num_tslots
        self.Dyn_gen_recalc = np.ones(nTS, dtype=bool)
        #np.ones(nTS, dtype=bool)
        self.Prop_recalc = np.ones(nTS, dtype=bool)
        self.Evo_init2t_recalc = np.ones(nTS + 1, dtype=bool)
        self.Evo_init2t_recalc[0] = False
        self.Evo_t2targ_recalc = np.ones(nTS + 1, dtype=bool)
        self.Evo_t2targ_recalc[-1] = False
        
        # The _calc_now map is used to during the calcs to specify
        # which values need updating immediately
        self.Dyn_gen_calc_now = np.zeros(nTS, dtype=bool)
        self.Prop_calc_now = np.zeros(nTS, dtype=bool)
        self.Evo_init2t_calc_now = np.zeros(nTS + 1, dtype=bool)
        self.Evo_t2targ_calc_now = np.zeros(nTS + 1, dtype=bool)
        
    def compare_amps(self, new_amps):
        """
        Determine which timeslots will have changed Hamiltonians
        i.e. any where control amplitudes have changed for that slot
        and mark (using masks) them and corresponding exponentiations and
        time evo operators for update
        Returns: True if amplitudes are the same, False if they have changed
        """
        dyn = self.parent
        nTS = dyn.num_tslots
        # create boolean array with same shape as ctrl_amps
        # True where value in New_amps differs, otherwise false
        if (self.parent.ctrl_amps == None):
            changedAmps = np.ones(new_amps.shape, dtype=bool)
        else:
            changedAmps = self.parent.ctrl_amps != new_amps
            
        if (self.msg_level >= 3):
            print "changedAmps"
            print changedAmps
        # create Boolean vector with same length as number of timeslots
        # True where any of the amplitudes have changed, otherwise false
        changedTSmask = np.any(changedAmps, 1)
        #print "changedTSmask"
        #print changedTSmask
        #if any of the amplidudes have changed then mark for recalc
        if (np.any(changedTSmask)):
            self.Dyn_gen_recalc[changedTSmask] = True
            self.Prop_recalc[changedTSmask] = True
            dyn.ctrl_amps = new_amps
            if (self.msg_level > 1):
                print "Control amplitudes updated"
            # find first and last changed Hamiltonians
            firstChangedHidx = None
            for i in range(nTS):
                if (changedTSmask[i]):
                    lastChangedHidx = i
                    if (firstChangedHidx == None):
                        firstChangedHidx = i
                    
            #set all fwd evo ops after first changed Ham to be recalculated
            self.Evo_init2t_recalc[firstChangedHidx + 1:] = True
            #set all bkwd evo ops up to (incl) last changed Ham to be recalculated
            self.Evo_t2targ_recalc[:lastChangedHidx + 1] = True
            
            # Flag fidelity and gradients as needing recalculation
            dyn.flag_system_changed()
            
            # *** update stats ***
            if (dyn.stats != None):
                dyn.stats.num_ctrl_amp_updates += 1
                dyn.stats.num_ctrl_amp_changes += changedAmps.sum()
                dyn.stats.num_timeslot_changes += changedTSmask.sum()
                
            return False
        else:
            return True

    def flag_all_calc_now(self):
        """
        Flags all Hamiltonians, propagators and propagations to be
        calculated now
        """
        # set flags for calculations
        self.Dyn_gen_calc_now[:] = True
        self.Prop_calc_now[:] = True
        self.Evo_init2t_calc_now[:-1] = True
        self.Evo_t2targ_calc_now[1:] = True
          
    def recompute_evolution(self):
        """
        Recalculates the Evo_init2t (forward) and Evo_t2targ (onward) time 
        evolution operators
        DynGen (Hamiltonians etc) and Prop (propagator) are calculated 
        as necessary 
        """
        if (self.msg_level >= 2):
            print "recomputing evolution (DynUpdate)"
            
        dyn = self.parent
        nTS = dyn.num_tslots
        # find the op slots that have been marked for update now
        # and need recalculation
        Evo_init2t_recomp_now = self.Evo_init2t_calc_now & \
                                            self.Evo_init2t_recalc
        Evo_t2targ_recomp_now = self.Evo_t2targ_calc_now & \
                                            self.Evo_t2targ_recalc
        
        # to recomupte Evo_init2t, will need to start
        #  at a cell that has been computed
        if (np.any(Evo_init2t_recomp_now)):
            for k in range(nTS, 0, -1):
                if Evo_init2t_recomp_now[k] and self.Evo_init2t_recalc[k-1]:
                    Evo_init2t_recomp_now[k-1] = True
                
        # for Evo_t2targ, will also need to start 
        #  at a cell that has been computed
        if (np.any(Evo_t2targ_recomp_now)):
            for k in range(0, nTS):
                if Evo_t2targ_recomp_now[k] and self.Evo_t2targ_recalc[k+1]:
                    Evo_t2targ_recomp_now[k+1] = True
                
        # determine which dyn gen and Prop need recalculating now in order to 
        # calculate the forwrd and onward evolutions
        Prop_recomp_now = (Evo_init2t_recomp_now[1:] \
                        | Evo_t2targ_recomp_now[:-1] \
                        | self.Prop_calc_now[:]) & self.Prop_recalc[:]
        Dyn_gen_recomp_now = (Prop_recomp_now[:] | self.Dyn_gen_calc_now[:]) \
                        & self.Dyn_gen_recalc[:]
                        
        if np.any(Dyn_gen_recomp_now):
            timeStart = timeit.default_timer()
            for k in range(nTS):
                if (Dyn_gen_recomp_now[k]):
                    # calculate the dynamics generators
                    dyn.Dyn_gen[k] = dyn.compute_dyn_gen(k)
                    self.Dyn_gen_recalc[k] = False
            if (dyn.stats != None):
                dyn.stats.num_dyn_gen_computes += Dyn_gen_recomp_now.sum()
                dyn.stats.wall_time_dyn_gen_compute += \
                                timeit.default_timer() - timeStart
                
        if np.any(Prop_recomp_now):
            timeStart = timeit.default_timer()
            for k in range(nTS):
                if (Prop_recomp_now[k]):
                    # calculate exp(H) and other per H computations needed for 
                    # the gradient function
                    dyn.Prop[k] = dyn.compute_propagator(k)
                    self.Prop_recalc[k] = False
            if (dyn.stats != None):
                dyn.stats.num_prop_computes += Prop_recomp_now.sum()
                dyn.stats.wall_time_prop_compute += \
                                    timeit.default_timer() - timeStart
        
        # compute the forward propagation
        if np.any(Evo_init2t_recomp_now):
            timeStart = timeit.default_timer()
            R = range(1, nTS + 1)
            for k in R:
                if (Evo_init2t_recomp_now[k]):
                    dyn.Evo_init2t[k] = \
                                    dyn.Prop[k-1].dot(dyn.Evo_init2t[k-1])
                    self.Evo_init2t_recalc[k] = False
            if (dyn.stats != None):
                dyn.stats.num_fwd_prop_step_computes += \
                                                    Evo_init2t_recomp_now.sum()
                dyn.stats.wall_time_fwd_prop_compute += \
                                            timeit.default_timer() - timeStart
                
        if np.any(Evo_t2targ_recomp_now):
            timeStart = timeit.default_timer()  
            # compute the onward propagation
            R = range(nTS-1, -1, -1)
            for k in R:
                if (Evo_t2targ_recomp_now[k]):
                    dyn.Evo_t2targ[k] = dyn.Evo_t2targ[k+1].dot(dyn.Prop[k])
                    self.Evo_t2targ_recalc[k] = False
            if (dyn.stats != None):
                dyn.stats.num_onwd_prop_step_computes += \
                                                Evo_t2targ_recomp_now.sum()
                dyn.stats.wall_time_onwd_prop_compute += \
                                            timeit.default_timer() - timeStart
                                                
        # Clear calc now flags
        self.Dyn_gen_calc_now[:] = False
        self.Prop_calc_now[:] = False
        self.Evo_init2t_calc_now[:] = False
        self.Evo_t2targ_calc_now[:] = False
        
    def get_timeslot_for_fidelity_calc(self):
        """
        Returns the timeslot index that will be used calculate current fidelity
        value. Attempts to find a timeslot where the least number of propagator
        calculations will be required.
        Flags the associated evolution operators for calculation now
        """
        dyn = self.parent
        nTS = dyn.num_tslots
        kBothEvoCurrent = -1
        kFwdEvoCurrent = -1
        kUse = -1
        # If no specific timeslot set in config, then determine dynamically
        if (kUse < 0):
            for k in range(nTS):
                # find first timeslot where both Evo_init2t and 
                # Evo_t2targ are current
                if (not self.Evo_init2t_recalc[k]):
                    kFwdEvoCurrent = k
                    if (not self.Evo_t2targ_recalc[k]):
                        kBothEvoCurrent = k
                        break
            
            if (kBothEvoCurrent >= 0):
                kUse = kBothEvoCurrent
            elif (kFwdEvoCurrent >= 0):
                kUse = kFwdEvoCurrent
            else:
                raise errors.FunctionalError(self.__class__.__name__ + \
                        ".get_current_value_setup_recalc", \
                        "No timeslot found matching criteria")
        
        self.Evo_init2t_calc_now[kUse] = True
        self.Evo_t2targ_calc_now[kUse] = True
        return kUse

# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]




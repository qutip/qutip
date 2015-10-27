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
Timeslot Computer
These classes determine which dynamics generators, propagators and evolutions
are recalculated when there is a control amplitude update.
The timeslot computer processes the lists held by the dynamics object

The default (UpdateAll) updates all of these each amp update, on the
assumption that all amplitudes are changed each iteration. This is typical
when using optimisation methods like BFGS in the GRAPE algorithm

The alternative (DynUpdate) assumes that only a subset of amplitudes
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
# QuTiP control modules
import qutip.control.errors as errors
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()


class TimeslotComputer:
    """
    Base class for all Timeslot Computers
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
        Note value should be set using set_log_level
    """
    def __init__(self, dynamics, params=None):
        self.parent = dynamics
        self.params = params
        self.reset()

    def reset(self):
        self.set_log_level(self.parent.log_level)
        self.id_text = 'TS_COMP_BASE'
        self._fwd_evo_tofh = 0
        self._owd_evo_tofh = 0
        self._prop_tofh = 0
        self._prop_grad_tofh = 0
        
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

    def flag_all_calc_now(self):
        pass

    def init_comp(self):
        pass

    def set_log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        self.log_level = lvl
        logger.setLevel(lvl)


class TSlotCompUpdateAll(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    def reset(self):
        TimeslotComputer.reset(self)
        self.id_text = 'ALL'
        self.apply_params()

    def compare_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        Returns: True if amplitudes are the same, False if they have changed
        """
        changed = False
        dyn = self.parent
        if dyn.ctrl_amps is None:
            # Flag fidelity and gradients as needing recalculation
            changed = True
            if dyn.stats is not None:
                dyn.stats.num_ctrl_amp_updates = 1
                dyn.stats.num_ctrl_amp_changes = len(new_amps.flat)
                dyn.stats.num_timeslot_changes = new_amps.shape[0]
        else:
            # create boolean array with same shape as ctrl_amps
            # True where value in new_amps differs, otherwise false
            changed_amps = dyn.ctrl_amps != new_amps
            if np.any(changed_amps):
                # Flag fidelity and gradients as needing recalculation
                changed = True
                if self.log_level <= logging.DEBUG:
                    logger.debug("{} amplitudes changed".format(
                        changed_amps.sum()))

                # *** update stats ***
                if dyn.stats is not None:
                    dyn.stats.num_ctrl_amp_updates += 1
                    dyn.stats.num_ctrl_amp_changes += changed_amps.sum()
                    changed_ts_mask = np.any(changed_amps, 1)
                    dyn.stats.num_timeslot_changes += changed_ts_mask.sum()
            else:
                if self.log_level <= logging.DEBUG:
                    logger.debug("No amplitudes changed")

        if changed:
            dyn.ctrl_amps = new_amps
            dyn.flag_system_changed()
            return False
        else:
            return True

    def recompute_evolution(self):
        """
        Recalculates the evolution operators.
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary
        """

        dyn = self.parent
        prop_comp = dyn.prop_computer
        n_ts = dyn.num_tslots
        n_ctrls = dyn.get_num_ctrls()

        if dyn.stats is not None:
            dyn.stats.num_tslot_recompute += 1
            if self.log_level <= logging.DEBUG:
                logger.log(logging.DEBUG, "recomputing evolution {} "
                           "(UpdateAll)".format(
                               dyn.stats.num_tslot_recompute))

        # calculate the Hamiltonians
        time_start = timeit.default_timer()
        for k in range(n_ts):
            dyn.dyn_gen[k] = dyn.combine_dyn_gen(k)
            if dyn.decomp_curr is not None:
                dyn.decomp_curr[k] = False
        if dyn.stats is not None:
            dyn.stats.wall_time_dyn_gen_compute += \
                timeit.default_timer() - time_start

        # calculate the propagators and the propagotor gradients
        if (dyn.config.test_out_prop or
                dyn.config.test_out_prop_grad or
                dyn.config.test_out_evo):
            f_ext = "_{}_{}_{}_call{}{}".format(dyn.id_text,
                                                dyn.prop_computer.id_text,
                                                dyn.fid_computer.id_text,
                                                dyn.stats.num_tslot_recompute,
                                                dyn.config.test_out_f_ext)

        if dyn.config.test_out_prop:
            fname = "prop" + f_ext
            fpath = os.path.join(dyn.config.test_out_dir, fname)
            self._prop_tofh = open(fpath, 'w')
        else:
            self._prop_tofh = 0

        if dyn.config.test_out_prop_grad:
            fname = "prop_grad" + f_ext
            fpath = os.path.join(dyn.config.test_out_dir, fname)
            self._prop_grad_tofh = open(fpath, 'w')
        else:
            self._prop_grad_tofh = 0

        time_start = timeit.default_timer()
        for k in range(n_ts):
            if prop_comp.grad_exact:
                for j in range(n_ctrls):
                    if j == 0:
                        prop, prop_grad = prop_comp.compute_prop_grad(k, j)
                        dyn.prop[k] = prop
                        dyn.prop_grad[k, j] = prop_grad
                        if self.log_level <= logging.DEBUG_INTENSE:
                            logger.log(logging.DEBUG_INTENSE,
                                       "propagator {}:\n{:10.3g}".format(
                                           k, prop))
                        if self._prop_tofh != 0:
                            self._prop_tofh.write(
                                "propagator k={}\n".format(k))
                            np.savetxt(self._prop_tofh, prop, fmt='%10.3g')

                    else:
                        prop_grad = prop_comp.compute_prop_grad(
                            k, j, compute_prop=False)
                        dyn.prop_grad[k, j] = prop_grad

                    if self._prop_grad_tofh != 0:
                        self._prop_grad_tofh.write(
                            "prop grad k={}, j={}\n".format(k, j))
                        np.savetxt(self._prop_grad_tofh, prop_grad,
                                   fmt='%10.3g')
            else:
                dyn.prop[k] = prop_comp.compute_propagator(k)
                if self._prop_tofh != 0:
                    self._prop_tofh.write(
                        "propagator k={}\n".format(k))
                    np.savetxt(self._prop_tofh, prop, fmt='%10.3g')

        if dyn.stats is not None:
            dyn.stats.wall_time_prop_compute += \
                timeit.default_timer() - time_start

        if self._prop_tofh != 0:
            self._prop_tofh.close()
            self._prop_tofh = 0
        if self._prop_grad_tofh != 0:
            self._prop_grad_tofh.close()
            self._prop_grad_tofh = 0

        # compute the forward propagation
        if dyn.config.test_out_evo:
            fname = "fwd_evo" + f_ext
            fpath = os.path.join(dyn.config.test_out_dir, fname)
            self._fwd_evo_tofh = open(fpath, 'w')

            fname = "owd_evo" + f_ext
            fpath = os.path.join(dyn.config.test_out_dir, fname)
            self._owd_evo_tofh = open(fpath, 'w')
        else:
            self._fwd_evo_tofh = 0
            self._owd_evo_tofh = 0

        time_start = timeit.default_timer()
        R = range(1, n_ts+1)
        for k in R:
            dyn.evo_init2t[k] = dyn.prop[k-1]*dyn.evo_init2t[k-1]
            if self._fwd_evo_tofh != 0:
                self._fwd_evo_tofh.write("Evo start to k={}\n".format(k))
                np.savetxt(self._fwd_evo_tofh, dyn.evo_init2t[k],
                           fmt='%10.3g')

        if dyn.stats is not None:
            dyn.stats.wall_time_fwd_prop_compute += \
                timeit.default_timer() - time_start

        time_start = timeit.default_timer()
        # compute the onward propagation
        if dyn.fid_computer.uses_evo_t2end:
            dyn.evo_t2end[n_ts - 1] = dyn.prop[n_ts - 1]
            R = range(n_ts-2, -1, -1)
            for k in R:
                dyn.evo_t2end[k] = dyn.evo_t2end[k+1]*dyn.prop[k]
                if self._fwd_evo_tofh != 0:
                    self._owd_evo_tofh.write("Evo k={} to end:\n".format(k))
                    np.savetxt(self._owd_evo_tofh, dyn.evo_t2end[k],
                               fmt='%14.6g')

        if dyn.fid_computer.uses_evo_t2targ:
            R = range(n_ts-1, -1, -1)
            for k in R:
                dyn.evo_t2targ[k] = dyn.evo_t2targ[k+1]*dyn.prop[k]
                if self._fwd_evo_tofh != 0:
                    self._owd_evo_tofh.write("Evo k={} to targ:\n".format(k))
                    np.savetxt(self._owd_evo_tofh, dyn.evo_t2targ[k],
                               fmt='%14.6g')

        if dyn.stats is not None:
            dyn.stats.wall_time_onwd_prop_compute += \
                timeit.default_timer() - time_start

        if self._fwd_evo_tofh != 0:
            self._fwd_evo_tofh.close()
            self._fwd_evo_tofh = 0
        if self._owd_evo_tofh != 0:
            self._owd_evo_tofh.close()
            self._owd_evo_tofh = 0

    def get_timeslot_for_fidelity_calc(self):
        """
        Returns the timeslot index that will be used calculate current fidelity
        value.
        This (default) method simply returns the last timeslot
        """
        return self.parent.num_tslots - 1


class TSlotCompDynUpdate(TimeslotComputer):
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
        self.dyn_gen_recalc = None
        self.prop_recalc = None
        self.evo_init2t_recalc = None
        self.evo_t2targ_recalc = None
        self.dyn_gen_calc_now = None
        self.prop_calc_now = None
        self.evo_init2t_calc_now = None
        self.evo_t2targ_calc_now = None
        TimeslotComputer.reset(self)
        self.id_text = 'DYNAMIC'
        self.apply_params()

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
        n_ts = self.parent.num_tslots
        self.dyn_gen_recalc = np.ones(n_ts, dtype=bool)
        # np.ones(n_ts, dtype=bool)
        self.prop_recalc = np.ones(n_ts, dtype=bool)
        self.evo_init2t_recalc = np.ones(n_ts + 1, dtype=bool)
        self.evo_init2t_recalc[0] = False
        self.evo_t2targ_recalc = np.ones(n_ts + 1, dtype=bool)
        self.evo_t2targ_recalc[-1] = False

        # The _calc_now map is used to during the calcs to specify
        # which values need updating immediately
        self.dyn_gen_calc_now = np.zeros(n_ts, dtype=bool)
        self.prop_calc_now = np.zeros(n_ts, dtype=bool)
        self.evo_init2t_calc_now = np.zeros(n_ts + 1, dtype=bool)
        self.evo_t2targ_calc_now = np.zeros(n_ts + 1, dtype=bool)

    def compare_amps(self, new_amps):
        """
        Determine which timeslots will have changed Hamiltonians
        i.e. any where control amplitudes have changed for that slot
        and mark (using masks) them and corresponding exponentiations and
        time evo operators for update
        Returns: True if amplitudes are the same, False if they have changed
        """
        dyn = self.parent
        n_ts = dyn.num_tslots
        # create boolean array with same shape as ctrl_amps
        # True where value in New_amps differs, otherwise false
        if self.parent.ctrl_amps is None:
            changed_amps = np.ones(new_amps.shape, dtype=bool)
        else:
            changed_amps = self.parent.ctrl_amps != new_amps

        if self.log_level <= logging.DEBUG_VERBOSE:
            logger.log(logging.DEBUG_VERBOSE, "changed_amps:\n{}".format(
                changed_amps))
        # create Boolean vector with same length as number of timeslots
        # True where any of the amplitudes have changed, otherwise false
        changed_ts_mask = np.any(changed_amps, 1)
        # if any of the amplidudes have changed then mark for recalc
        if np.any(changed_ts_mask):
            self.dyn_gen_recalc[changed_ts_mask] = True
            self.prop_recalc[changed_ts_mask] = True
            dyn.ctrl_amps = new_amps
            if self.log_level <= logging.DEBUG:
                logger.debug("Control amplitudes updated")
            # find first and last changed dynamics generators
            first_changed = None
            for i in range(n_ts):
                if changed_ts_mask[i]:
                    last_changed = i
                    if first_changed is None:
                        first_changed = i

            # set all fwd evo ops after first changed Ham to be recalculated
            self.evo_init2t_recalc[first_changed + 1:] = True
            # set all bkwd evo ops up to (incl) last changed Ham to be
            # recalculated
            self.evo_t2targ_recalc[:last_changed + 1] = True

            # Flag fidelity and gradients as needing recalculation
            dyn.flag_system_changed()

            # *** update stats ***
            if dyn.stats is not None:
                dyn.stats.num_ctrl_amp_updates += 1
                dyn.stats.num_ctrl_amp_changes += changed_amps.sum()
                dyn.stats.num_timeslot_changes += changed_ts_mask.sum()

            return False
        else:
            return True

    def flag_all_calc_now(self):
        """
        Flags all Hamiltonians, propagators and propagations to be
        calculated now
        """
        # set flags for calculations
        self.dyn_gen_calc_now[:] = True
        self.prop_calc_now[:] = True
        self.evo_init2t_calc_now[:-1] = True
        self.evo_t2targ_calc_now[1:] = True

    def recompute_evolution(self):
        """
        Recalculates the evo_init2t (forward) and evo_t2targ (onward) time
        evolution operators
        DynGen (Hamiltonians etc) and prop (propagator) are calculated
        as necessary
        """
        if self.log_level <= logging.DEBUG_VERBOSE:
            logger.log(logging.DEBUG_VERBOSE, "recomputing evolution "
                       "(DynUpdate)")

        dyn = self.parent
        n_ts = dyn.num_tslots
        # find the op slots that have been marked for update now
        # and need recalculation
        evo_init2t_recomp_now = self.evo_init2t_calc_now & \
            self.evo_init2t_recalc
        evo_t2targ_recomp_now = self.evo_t2targ_calc_now & \
            self.evo_t2targ_recalc

        # to recomupte evo_init2t, will need to start
        #  at a cell that has been computed
        if np.any(evo_init2t_recomp_now):
            for k in range(n_ts, 0, -1):
                if evo_init2t_recomp_now[k] and self.evo_init2t_recalc[k-1]:
                    evo_init2t_recomp_now[k-1] = True

        # for evo_t2targ, will also need to start
        #  at a cell that has been computed
        if np.any(evo_t2targ_recomp_now):
            for k in range(0, n_ts):
                if evo_t2targ_recomp_now[k] and self.evo_t2targ_recalc[k+1]:
                    evo_t2targ_recomp_now[k+1] = True

        # determine which dyn gen and prop need recalculating now in order to
        # calculate the forwrd and onward evolutions
        prop_recomp_now = (evo_init2t_recomp_now[1:]
                           | evo_t2targ_recomp_now[:-1]
                           | self.prop_calc_now[:]) & self.prop_recalc[:]
        dyn_gen_recomp_now = (prop_recomp_now[:] | self.dyn_gen_calc_now[:]) \
            & self.dyn_gen_recalc[:]

        if np.any(dyn_gen_recomp_now):
            time_start = timeit.default_timer()
            for k in range(n_ts):
                if dyn_gen_recomp_now[k]:
                    # calculate the dynamics generators
                    dyn.dyn_gen[k] = dyn.compute_dyn_gen(k)
                    self.dyn_gen_recalc[k] = False
            if dyn.stats is not None:
                dyn.stats.num_dyn_gen_computes += dyn_gen_recomp_now.sum()
                dyn.stats.wall_time_dyn_gen_compute += \
                    timeit.default_timer() - time_start

        if np.any(prop_recomp_now):
            time_start = timeit.default_timer()
            for k in range(n_ts):
                if prop_recomp_now[k]:
                    # calculate exp(H) and other per H computations needed for
                    # the gradient function
                    dyn.prop[k] = dyn.compute_propagator(k)
                    self.prop_recalc[k] = False
            if dyn.stats is not None:
                dyn.stats.num_prop_computes += prop_recomp_now.sum()
                dyn.stats.wall_time_prop_compute += \
                    timeit.default_timer() - time_start

        # compute the forward propagation
        if np.any(evo_init2t_recomp_now):
            time_start = timeit.default_timer()
            R = range(1, n_ts + 1)
            for k in R:
                if evo_init2t_recomp_now[k]:
                    dyn.evo_init2t[k] = \
                        dyn.prop[k-1].dot(dyn.evo_init2t[k-1])
                    self.evo_init2t_recalc[k] = False
            if dyn.stats is not None:
                dyn.stats.num_fwd_prop_step_computes += \
                    evo_init2t_recomp_now.sum()
                dyn.stats.wall_time_fwd_prop_compute += \
                    timeit.default_timer() - time_start

        if np.any(evo_t2targ_recomp_now):
            time_start = timeit.default_timer()
            # compute the onward propagation
            R = range(n_ts-1, -1, -1)
            for k in R:
                if evo_t2targ_recomp_now[k]:
                    dyn.evo_t2targ[k] = dyn.evo_t2targ[k+1].dot(dyn.prop[k])
                    self.evo_t2targ_recalc[k] = False
            if dyn.stats is not None:
                dyn.stats.num_onwd_prop_step_computes += \
                    evo_t2targ_recomp_now.sum()
                dyn.stats.wall_time_onwd_prop_compute += \
                    timeit.default_timer() - time_start

        # Clear calc now flags
        self.dyn_gen_calc_now[:] = False
        self.prop_calc_now[:] = False
        self.evo_init2t_calc_now[:] = False
        self.evo_t2targ_calc_now[:] = False

    def get_timeslot_for_fidelity_calc(self):
        """
        Returns the timeslot index that will be used calculate current fidelity
        value. Attempts to find a timeslot where the least number of propagator
        calculations will be required.
        Flags the associated evolution operators for calculation now
        """
        dyn = self.parent
        n_ts = dyn.num_tslots
        kBothEvoCurrent = -1
        kFwdEvoCurrent = -1
        kUse = -1
        # If no specific timeslot set in config, then determine dynamically
        if kUse < 0:
            for k in range(n_ts):
                # find first timeslot where both evo_init2t and
                # evo_t2targ are current
                if not self.evo_init2t_recalc[k]:
                    kFwdEvoCurrent = k
                    if not self.evo_t2targ_recalc[k]:
                        kBothEvoCurrent = k
                        break

            if kBothEvoCurrent >= 0:
                kUse = kBothEvoCurrent
            elif kFwdEvoCurrent >= 0:
                kUse = kFwdEvoCurrent
            else:
                raise errors.FunctionalError("No timeslot found matching "
                                             "criteria")

        self.evo_init2t_calc_now[kUse] = True
        self.evo_t2targ_calc_now[kUse] = True
        return kUse

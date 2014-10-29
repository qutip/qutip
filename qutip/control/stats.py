# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:18:14 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Statistics for the optimisation
Note that some of the stats here are redundant copies from the optimiser
used here for calculations
"""
#import numpy as np
import datetime

class Stats:
    """
    Base class for all optimisation statistics
    Used for configurations where all timeslots are updated each iteration
    e.g. exact gradients
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        # Start stop and total wall time for the optimisation
        self.wall_time_optim_start = 0.0
        self.wall_time_optim_end = 0.0
        self.wall_time_optim = 0.0
        # Number of iterations of the optimisation algorithm (e.g. BFGS)
        self.num_iter = 0
        
        # total wall (elasped) time computing Hamiltonians
        self.wall_time_dyn_gen_compute = 0.0        
        # total wall (elasped) time computing propagators
        self.wall_time_prop_compute = 0.0
        # total wall (elasped) time computing forward propagation
        self.wall_time_fwd_prop_compute = 0.0
        # total wall (elasped) time computing onward propagation
        self.wall_time_onwd_prop_compute = 0.0
        
        # Number of calls to fidelity function by the optimisation algorithm
        self.num_fidelity_func_calls = 0
        # Number of calls to gradient function by the optimisation algorithm
        self.num_grad_func_calls = 0
        # Number of time the fidelity is computed
        self.num_fidelity_computes = 0
        # Number of time the gradient is computed
        self.num_grad_computes = 0
        # **** Control amplitudes *****
        # Number of times the control amplitudes are updated
        self.num_ctrl_amp_updates = 0
        # Mean number of ontrol amplitude updates per iteration
        self.mean_num_ctrl_amp_updates_per_iter = 0.0
        # Number of times individual control amplitudes are changed
        self.num_timeslot_changes = 0
        # Mean average number of control amplitudes that are changed per update
        self.mean_num_timeslot_changes_per_update = 0.0
        # Number of times individual control amplitudes are changed
        self.num_ctrl_amp_changes = 0
        # Mean average number of control amplitudes that are changed per update
        self.mean_num_ctrl_amp_changes_per_update = 0.0
        
        self.wall_time_gradient_compute = 0.0
        
    def calculate(self):
        """
        Perform and calculations (e.g. averages) that are required on the stats
        Should be called before calling report
        """
        # If the optimation is still running then the optimisation 
        # time is the time so far
        if (self.wall_time_optim_end > 0.0):
            self.wall_time_optim = \
                    self.wall_time_optim_end - self.wall_time_optim_start
                    
        self.mean_num_ctrl_amp_updates_per_iter = \
                self.num_ctrl_amp_updates / float(self.num_iter)
        
        self.mean_num_timeslot_changes_per_update = \
                self.num_timeslot_changes / float(self.num_ctrl_amp_updates)
                
        self.mean_num_ctrl_amp_changes_per_update = \
                self.num_ctrl_amp_changes / float(self.num_ctrl_amp_updates)               

    def _format_datetime(self, t, tot=0.0):
        dtStr = str(datetime.timedelta(seconds=t))
        if (tot > 0):
            percent = 100*t/tot
            dtStr += " ({:03.2f}%)".format(percent)
        return dtStr
    def report(self):
        """
        Print a report of the stats to the console
        """
        print ""
        print "------------------------------------"
        print "---- Control optimisation stats ----"
        self.report_timings()
        self.report_func_calls()
        self.report_amp_updates()
        print "---------------------------"
        
    
    def report_timings(self):
        print "**** Timings (HH:MM:SS.US) ****"
        tot = self.wall_time_optim
        print "Total wall time elapsed during optimisation: " + \
                self._format_datetime(tot)
        print "Wall time computing Hamiltonians: " + \
                self._format_datetime(self.wall_time_dyn_gen_compute, tot)
        print "Wall time computing propagators: " + \
                self._format_datetime(self.wall_time_prop_compute, tot)
        print "Wall time computing forward propagation: " + \
                self._format_datetime(self.wall_time_fwd_prop_compute, tot)
        print "Wall time computing onward propagation: " + \
                self._format_datetime(self.wall_time_onwd_prop_compute, tot)
        print "Wall time computing gradient: " + \
                self._format_datetime(self.wall_time_gradient_compute, tot)
        print ""
        
    def report_func_calls(self):
        print "**** Iterations and function calls ****"
        print "Number of iterations: " + str(self.num_iter)
        print "Number of fidelity function calls: " + \
                str(self.num_fidelity_func_calls)
        print "Number of times fidelity is computed: " + \
                str(self.num_fidelity_computes)
        print "Number of gradient function calls: " + \
                str(self.num_grad_func_calls)
        print "Number of times gradients are computed: " + \
                str(self.num_grad_computes)
        print ""
        
    def report_amp_updates(self):
        print "**** Control amplitudes ****"
        print "Number of control amplitude updates: " + \
                str(self.num_ctrl_amp_updates)
        print "Mean number of updates per iteration: " + \
                str(self.mean_num_ctrl_amp_updates_per_iter)                
        print "Number of timeslot values changed: " + \
                str(self.num_timeslot_changes)
        print "Mean number of timeslot changes per update: " + \
                str(self.mean_num_timeslot_changes_per_update)
        print "Number of amplitude values changed: " + \
                str(self.num_ctrl_amp_changes)
        print "Mean number of amplitude changes per update: " + \
                str(self.mean_num_ctrl_amp_changes_per_update)
                

class Stats_DynTsUpdate(Stats):
    """
    Optimisation stats class for configurations where all timeslots are not  
    necessarily updated at each iteration. In this case it may be interesting
    to know how many Hamiltions etc are computed each ctrl amplitude update 
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        Stats.reset(self)
        # **** Hamiltonians *****        
        # Total number of Hamiltonian computations
        self.num_dyn_gen_computes = 0
        # Mean average number of Hamiltonian computations per update
        self.mean_num_dyn_gen_computes_per_update = 0.0
        # Mean average time to compute a Hamiltonian
        self.mean_wall_time_dyn_gen_compute = 0.0
        # **** Propagators *****
        # Total number of propagator computations
        self.num_prop_computes = 0
        # Mean average number of propagator computations per update
        self.mean_num_prop_computes_per_update = 0.0
        # Mean average time to compute a propagator
        self.mean_wall_time_prop_compute = 0.0
        # **** Forward propagation ****
        # Total number of steps (inner products) computing forward propagation
        self.num_fwd_prop_step_computes = 0
        # Mean average number of steps computing forward propagation
        self.mean_num_fwd_prop_step_computes_per_update = 0.0
        # Mean average time to compute forward propagation
        self.mean_wall_time_fwd_prop_compute = 0.0
        # **** onward propagation ****
        # Total number of steps (inner products) computing onward propagation
        self.num_onwd_prop_step_computes = 0
        # Mean average number of steps computing onward propagation
        self.mean_num_onwd_prop_step_computes_per_update = 0.0
        # Mean average time to compute onward propagation
        self.mean_wall_time_onwd_prop_compute = 0.0
        
    def calculate(self):
        Stats.calculate(self)
        self.mean_num_dyn_gen_computes_per_update = \
                self.num_dyn_gen_computes / float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_dyn_gen_compute = \
                self.wall_time_dyn_gen_compute / \
                        float(self.num_dyn_gen_computes)

        self.mean_num_prop_computes_per_update = \
                self.num_prop_computes / float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_prop_compute = \
                self.wall_time_prop_compute / float(self.num_prop_computes)
                
        self.mean_num_fwd_prop_step_computes_per_update = \
                self.num_fwd_prop_step_computes / \
                        float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_fwd_prop_compute = \
                self.wall_time_fwd_prop_compute / \
                        float(self.num_fwd_prop_step_computes)
    
        self.mean_num_onwd_prop_step_computes_per_update = \
                self.num_onwd_prop_step_computes / \
                        float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_onwd_prop_compute = \
                self.wall_time_onwd_prop_compute / \
                        float(self.num_onwd_prop_step_computes)
        
    def report(self):
        """
        Print a report of the stats to the console
        """
        print ""
        print "------------------------------------"
        print "---- Control optimisation stats ----"
        self.report_timings()
        self.report_func_calls()
        self.report_amp_updates()
        self.report_dyn_gen_comps()
        self.report_fwd_prop()
        self.report_onwd_prop()
        print "---------------------------"
        
    def report_dyn_gen_comps(self):
        print "**** Hamiltonian Computations ****"
        print "Total number of Hamiltonian computations: " + \
                str(self.num_dyn_gen_computes)        
        print "Mean number of Hamiltonian computations per update: " + \
                str(self.mean_num_dyn_gen_computes_per_update)
        print "Mean wall time to compute Hamiltonian {} s".\
                format(self.mean_wall_time_dyn_gen_compute)               
        print "**** Propagator Computations ****"
        print "Total number of propagator computations: " + \
                str(self.num_prop_computes)        
        print "Mean number of propagator computations per update: " + \
                str(self.mean_num_prop_computes_per_update)
        print "Mean wall time to compute propagator {} s".\
                format(self.mean_wall_time_prop_compute)
                
    def report_fwd_prop(self):
        print "**** Forward Propagation ****"
        print "Total number of forward propagation step computations: " + \
                str(self.num_fwd_prop_step_computes)        
        print "Mean number of forward propagation step computations" + \
                " per update: " + \
                str(self.mean_num_fwd_prop_step_computes_per_update)
        print "Mean wall time to compute forward propagation {} s".\
                format(self.mean_wall_time_fwd_prop_compute)
                
    def report_onwd_prop(self):
        print "**** Onward Propagation ****"
        print "Total number of onward propagation step computations: " + \
                str(self.num_onwd_prop_step_computes)        
        print "Mean number of onward propagation step computations" + \
                " per update: " + \
                str(self.mean_num_onwd_prop_step_computes_per_update)
        print "Mean wall time to compute onward propagation {} s".\
                format(self.mean_wall_time_onwd_prop_compute)    
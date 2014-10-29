# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:14:58 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing.

Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.optimize_pulse function.
The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which in this case is given by the
'Trace difference' norm.

This in an Symplectic quantum system example, with two coupled oscillators

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""

import numpy as np
import numpy.matlib as mat
import scipy.linalg as la
import matplotlib.pyplot as plt
import datetime

import qutip.control.ctrlpulseoptim as cpo
import qutip.control.symplectic as sympl

example_name = 'Symplectic'
msg_level = 1

# ****************************************************************
# Define the physics of the problem

w1 = 1
w2 = 1
g1 = 0.5
A0 = np.array([[w1, 0, g1, 0], \
                [0, w1, 0, g1], \
                [g1, 0, w2, 0], \
                [0, g1, 0, w2]])
    
Ac = np.array([[1, 0, 0, 0,], \
                [0, 1, 0, 0], \
                [0, 0, 0, 0], \
                [0, 0, 0, 0]])
Ctrls = [Ac]        
n_ctrls = len(Ctrls)
initial = mat.eye(4)

# Goal
a = 1
Ag = np.array([[0, 0, a, 0], \
                [0, 0, 0, a], \
                [a, 0, 0, 0], \
                [0, a, 0, 0]])
               
Sg = la.expm(sympl.calc_omega(2).dot(Ag))

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 1000
# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 30
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-8


# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'ZERO'
# *************************************************************
# File extension for output files

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation
print ""
print "***********************************"
print "Starting pulse optimisation"
# Note that this call uses
#    dyn_type='SYMPL'
# This means that matrices that describe the dynamics are assumed to be
# Symplectic, i.e. the propagator can be calculated using 
# expm(combined_dynamics.omega*dt)
# This has defaults for:
#    prop_type='FRECHET'
# therefore the propagators and their gradients will be calculated using the
# Frechet method, i.e. an exact gradent
#    fid_type='TRACEDIFF'
# so that the fidelity error, i.e. distance from the target, is give
# by the trace of the difference between the target and evolved operators 
result = cpo.optimize_pulse(A0, Ctrls, initial, Ag, n_ts, evo_time, \
                fid_err_targ=fid_err_targ, min_grad=min_grad, \
                max_iter=max_iter, max_wall_time=max_wall_time, \
                dyn_type='SYMPL', \
                out_file_ext=f_ext, init_pulse_type=p_type, \
                msg_level=2, gen_stats=True)

print ""
print "***********************************"
print "Optimising complete. Stats follow:"
result.stats.report()
print ""
print "Final evolution"
print result.evo_full_final
print ""

print "********* Summary *****************"
print "Final fidelity error {}".format(result.fid_err)
print "Terminated due to {}".format(result.termination_reason)
print "Number of iterations {}".format(result.num_iter)
#print "wall time: ", result.wall_time
print "Completed in {} HH:MM:SS.US".\
        format(datetime.timedelta(seconds=result.wall_time))
# print "Final gradient normal {}".format(result.grad_norm_final)
print "***********************************"

# Plot the initial and final amplitudes
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial ctrl amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
t = result.time[:n_ts]
for j in range(n_ctrls):
    amps = result.initial_amps[:, j]
    ax1.plot(t, amps)
ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    amps = result.final_amps[:, j]
    ax2.plot(t, amps)

plt.show()


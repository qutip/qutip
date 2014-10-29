# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:58:53 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.optimize_pulse_unitary function.
The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which is equivalent maximising the fidelity
to optimal value of 1.

The system in this example is a single qubit in a constant field in z
with a variable control field in x
The target evolution is the Hadamard gate irrespective of global phase

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""

import numpy as np
import numpy.matlib as mat
import matplotlib.pyplot as plt
import datetime

import qutip.control.ctrlpulseoptim as cpo

example_name = 'Hadamard'
msg_level = 1
# ****************************************************************
# Define the physics of the problem

nSpins = 1
Sx = np.array([[0, 1], [1, 0]], dtype=complex)
Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = np.array([[1, 0], [0, -1]], dtype=complex)
Si = mat.eye(2)/2

H_d = Sz
H_c = [Sx]
# Number of ctrls
n_ctrls = len(H_c)

U_0 = mat.eye(2**nSpins)
# Hadamard gate
U_targ = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 500
# Time allowed for the evolution
evo_time = 10

# ***** Define the termination conditions *****
# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20


# Initial pulse type
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'
# *************************************************************
# File extension for output files

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation
print ""
print "***********************************"
print "Starting pulse optimisation"
result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, \
                fid_err_targ=fid_err_targ, min_grad=min_grad, \
                max_iter=max_iter, max_wall_time=max_wall_time, \
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

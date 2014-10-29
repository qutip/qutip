# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:18:29 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.create_pulse_optimizer function to 
generate an Optimizer object, through which the configuration can be
manipulated before running the optmisation algorithm. In this case it is
demonstrated by modifying the initial ctrl pulses.

The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which is equivalent maximising the fidelity
to optimal value of 1.

The system in this example is two qubits in constant fields in x, y and z
with a variable independant controls fields in x and y acting on each qubit
The target evolution is the QFT gate. The user can experiment with the
different:
    phase options - phase_option = SU or PSU
    propagtor computer type prop_type = DIAG or FRECHET
    fidelity measures - fid_type = UNIT or TRACEDIFF

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot
"""

import numpy as np
import numpy.matlib as mat
from numpy.matlib import kron
import matplotlib.pyplot as plt
import datetime

import qutip.control.ctrlpulseoptim as cpo
import qutip.control.pulsegen as pulsegen
from qutip.control import gates

example_name = 'QFT'
msg_level = 1
# ****************************************************************
# Define the physics of the problem
nSpins = 2
Sx = np.array([[0, 1], [1, 0]], dtype=complex)
Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = np.array([[1, 0], [0, -1]], dtype=complex)
Si = mat.eye(2)/2

H_d = 0.5*(kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz))
H_c = [kron(Sx, Si), kron(Sy, Si), kron(Si, Sx), kron(Si, Sy)]
n_ctrls = len(H_c)

U_0 = mat.eye(2**nSpins)

# Quantum Fourier Transform gate
# This produces values with small rounding errors
U_targ = gates.qft(nSpins)
# can use this for nSpins=2
U_targ = np.array([
    [0.5+0.0j,  0.5+0.0j,  0.5+0.0j,  0.5+0.0j],
    [0.5+0.0j,  0.0+0.5j, -0.5+0.0j, -0.0-0.5j],
    [0.5+0.0j, -0.5+0.0j,  0.5+0.0j, -0.5+0.0j],
    [0.5+0.0j,  0.0-0.5j, -0.5+0.0j,  0.0+0.5j],
    ])

# ***** Define time evolution parameters *****
# Number of time slots
n_ts = 100
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
p_type = 'LIN'
# *************************************************************
# File extension for output files

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Run the optimisation
print ""
print "***********************************"
print "Creating optimiser objects"
optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, n_ts, evo_time, \
                amp_lbound=-5.0, amp_ubound=5.0, \
                fid_err_targ=fid_err_targ, min_grad=min_grad, \
                max_iter=max_iter, max_wall_time=max_wall_time, \
                optim_alg='LBFGSB', dyn_type='UNIT', \
                prop_type='DIAG', fid_type='UNIT', phase_option='PSU', \
                init_pulse_type=p_type, \
                msg_level=2, gen_stats=True)
                
print ""
print "***********************************"
print "Configuring optimiser objects"
# **** Set some optimiser config parameters ****
# Increase the number of gradient values used to estimate the Hessian
# (note this only affects LBFGS)
optim.config.max_metric_corr = 20
# Decrease the 'accuracy factor' of the algorithm
# (note this only affects LBFGS)
optim.config.optim_alg_acc_fact = 1e8

dyn = optim.dynamics
# Generate different pulses for each control
p_gen = optim.pulse_generator
init_amps = np.zeros([n_ts, n_ctrls])
if (p_gen.periodic):
    phase_diff = np.pi / n_ctrls
    for j in range(n_ctrls):
        init_amps[:, j] = p_gen.gen_pulse(start_phase=phase_diff*j)
elif (isinstance(p_gen, pulsegen.PulseGen_Linear)):
    for j in range(n_ctrls):
        p_gen.scaling = float(j) - float(n_ctrls - 1)/2
        init_amps[:, j] = p_gen.gen_pulse()
elif (isinstance(p_gen, pulsegen.PulseGen_Zero)):
    for j in range(n_ctrls):
        p_gen.offset = sf = float(j) - float(n_ctrls - 1)/2
        init_amps[:, j] = p_gen.gen_pulse()
else:
    # Should be random pulse
    for j in range(n_ctrls):
        init_amps[:, j] = p_gen.gen_pulse()

dyn.initialize_controls(init_amps)

# Save initial amplitudes to a text file
pulsefile = "ctrl_amps_initial_" + f_ext
dyn.save_amps(pulsefile)
if (msg_level >= 1):
    print "Initial amplitudes output to file: " + pulsefile

print "***********************************"
print "Starting pulse optimisation"
result = optim.run_optimization()

# Save final amplitudes to a text file
pulsefile = "ctrl_amps_final_" + f_ext
dyn.save_amps(pulsefile)
if (msg_level >= 1):
    print "Final amplitudes output to file: " + pulsefile
        
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



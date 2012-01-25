#
# Test program for comparing the Lindblad and Bloch-Redfield master equations
# 
from qutip import *
from pylab import *
import time

def qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist):
    #
    # Hamiltonian
    #
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    H = w * (cos(theta) * sz + sin(theta) * sx)
    #H = w * sz

    #
    # Lindblad master equation
    #
    c_op_list = []
    n_th = 0.0 # zero temperature
    rate = gamma1 * (n_th + 1)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())
    lme_results = odesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])  

    #
    # Bloch-Redfield tensor
    #
    ohmic_spectrum = lambda w: 0.1*gamma1 * w / (2*pi) * (w > 0.0)    
    R = bloch_redfield_tensor(H, [sx], [ohmic_spectrum])
    brme_results = brmesolve(R, psi0, tlist, [sx, sy, sz])   

    return lme_results, brme_results
    
#
# set up the calculation
#
w     = 1.0 * 2 * pi  # qubit angular frequency
theta = 0.5 * pi      # qubit angle from sigma_z axis (toward sigma_x axis)
gamma1 = 0.5          # qubit relaxation rate
gamma2 = 0.2          # qubit dephasing rate
# initial state
a = 1.0
psi0 = (a* basis(2,0) + (1-a)*basis(2,1))/(sqrt(a**2 + (1-a)**2))
tlist = linspace(0,15,1000)
start_time = time.time()
lme_results, brme_results = qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist)
stop_time = time.time()
print 'time elapsed: '+str(stop_time-start_time)

fig = figure()
ax = fig.add_subplot(2,1,1)
title('Lindblad master equation')
ax.plot(tlist, lme_results[0], 'r')
ax.plot(tlist, lme_results[1], 'g')
ax.plot(tlist, lme_results[2], 'b')
ax.legend(("sx", "sy", "sz"))

ax = fig.add_subplot(2,1,2)
title('Bloch-Redfield master equation')
ax.plot(tlist, brme_results[0], 'r')
ax.plot(tlist, brme_results[1], 'g')
ax.plot(tlist, brme_results[2], 'b')
ax.legend(("sx", "sy", "sz"))

show()

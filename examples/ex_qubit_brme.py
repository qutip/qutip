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
    print "H =\n", H

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

    L = liouvillian(H, c_op_list)
    print "L.re =\n", real(L.full())
    print "L.im =\n", imag(L.full())


    #
    # Bloch-Redfield tensor
    #
    #ohmic_spectrum = lambda w: gamma1 * w / (2*pi)**2 * (w > 0.0)    
    ohmic_spectrum = lambda w: gamma1 * w / (2*pi)  
    R, ekets = bloch_redfield_tensor(H, [sx], [ohmic_spectrum])
    
    print "R.re =\n", real(R.full())
    print "R.im =\n", imag(R.full())
        
    brme_results = brmesolve(R, ekets, psi0, tlist, [sx, sy, sz])   

    return lme_results, brme_results
    
#
# set up the calculation
#
w     = 1.0 * 2 * pi  # qubit angular frequency
theta = 0.0 * pi      # qubit angle from sigma_z axis (toward sigma_x axis)
gamma1 = 0.2          # qubit relaxation rate
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

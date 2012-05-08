#
# Find the steady state of a strongly driven qubit as a function of 
# driving amplitude and qubit bias. 
#
# Note: This calculation takes a long time.
#
from qutip import *
import time

from pylab import *

def hamiltonian_t(t, args):
    #
    # evaluate the hamiltonian at time t. 
    #
    H0 = args[0]
    H1 = args[1]
    w  = args[2]

    return H0 + H1 * sin(w * t)

def sd_qubit_integrate(delta, eps0_vec, A_vec, w, gamma1, gamma2):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    # collapse operators
    c_op_list = []

    n_th = 0.0 # zero temperature

    # relaxation
    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    # excitation
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())

    # dephasing 
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)


    N = len(A_vec)
    M = len(eps0_vec)
    p_ex = zeros([N, M]) #, dtype=complex)

    T = 2 * pi / w

    sn = sm.dag() * sm

    # sweep over the driving amplitude and bias point, find the steady state 
    # for each point and store in a matrix
    for n in range(0, N):
        for m in range(0, M):

            H0 = - delta/2.0 * sx - eps0_vec[m]/2.0 * sz
            H1 = - A_vec[n] * sx
        
            H_args = (H0, H1, w)

            # find the propagator for one period of the time-dependent
            # hamiltonian
            U = propagator(hamiltonian_t, T, c_op_list, H_args)

            # find the steady state of the driven system 
            rho_ss = propagator_steadystate(U)
        
            p_ex[n, m] = real(expect(sn, rho_ss))

        print "Percent completed: ", (100.0 * (n+1)) / N

    return p_ex
    
#
# set up the calculation
#
delta = 0.2  * 2 * pi   # qubit sigma_x coefficient
w     = 1.0  * 2 * pi   # qubit sigma_z coefficient

A_vec    = arange(0.0, 4.0, 0.025) * 2 * pi  # driving amplitude
eps0_vec = arange(0.0, 4.0, 0.025) * 2 * pi  # qubit sigma-z bias point

gamma1 = 0.05          # relaxation rate
gamma2 = 0.0           # dephasing  rate

start_time = time.time()
p_ex = sd_qubit_integrate(delta, eps0_vec, A_vec, w, gamma1, gamma2)
print 'time elapsed = ' + str(time.time() - start_time) 

figure(1)
pcolor(A_vec, eps0_vec, real(p_ex), edgecolors='none')
xlabel('A/w')
ylabel('eps0/w')
title('Excitation probabilty of qubit, in steady state')
show()



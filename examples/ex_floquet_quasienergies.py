#
# Example: Find the floquet modes and quasi energies for a driven system and
# plot the floquet states/quasienergies for one period of the driving.
#
from qutip import *
from pylab import *
import time

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args['H0']
    H1 = args['H1']
    w  = args['w']

    return H0 + sin(w * t) * H1

def H1_coeff_t(t, args):
    return sin(args['w'] * t)

def qubit_integrate(delta, eps0_vec, A, omega, gamma1, gamma2, psi0, T, option):

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


    #quasi_energies = zeros((len(A_vec), 2))
    #f_gnd_prob     = zeros((len(A_vec), 2))
    quasi_energies = zeros((len(eps0_vec), 2))
    f_gnd_prob     = zeros((len(eps0_vec), 2))
    wf_gnd_prob    = zeros((len(eps0_vec), 2))

    idx = 0
    #for A in A_vec:
    for eps0 in eps0_vec:

        H0 = - delta/2.0 * sx - eps0/2.0 * sz
        H1 = A/2.0 * sz
       
        # H = H0 + H1 * sin(w * t) in the 'list-string' format
        H = [H0, [H1, 'sin(w * t)']]
        #Hargs = {'H0': H0, 'H1': H1, 'w': omega}
        #H = [H0, [H1, H1_coeff_t]]
        Hargs = {'w': omega}
            
        # find the propagator for one driving period
        f_modes,f_energies = floquet_modes(H, T, Hargs)
        #f_modes,f_energies = floquet_modes(hamiltonian_t, T, Hargs)

        print "Floquet quasienergies[",idx,"] =", f_energies

        quasi_energies[idx,:] = f_energies

        #print "Floquet state 0 =", f_states_0[0]
        #print "Floquet state 1 =", f_states_0[1]        

        f_gnd_prob[idx, 0] = expect(sm.dag() * sm, f_modes[0])
        f_gnd_prob[idx, 1] = expect(sm.dag() * sm, f_modes[1])

        f_states = floquet_states_t(f_modes, f_energies, 0, H, T, Hargs)
        #f_states = floquet_states_t(f_modes, f_energies, 0, hamiltonian_t, T, Hargs)

        wf_gnd_prob[idx, 0] = expect(sm.dag() * sm, f_states[0])
        wf_gnd_prob[idx, 1] = expect(sm.dag() * sm, f_states[1])

        #print "Floquet initial states =", f_states_0

        idx += 1
        
    return quasi_energies, f_gnd_prob, wf_gnd_prob
    
#
# set up the calculation: a strongly driven two-level system
# (repeated LZ transitions)
#
delta = 0.2 * 2 * pi  # qubit sigma_x coefficient
eps0  = 0.5 * 2 * pi  # qubit sigma_z coefficient
gamma1 = 0.0        # relaxation rate
gamma2 = 0.0         # dephasing  rate
A      = 2.0 * 2 * pi 
psi0   = basis(2,0)    # initial state
omega  = 1.0 * 2 * pi # driving frequency
T      = (2*pi)/omega  # driving period

param  = linspace(-5.0, 5.0, 200) * 2 * pi 

eps0 = param


start_time = time.time()
q_energies, f_gnd_prob, wf_gnd_prob = qubit_integrate(delta, eps0, A, omega, gamma1, gamma2, psi0, T, "dynamics")
print 'dynamics: time elapsed = ' + str(time.time() - start_time) 


#
# plot the results
#
figure(1)
plot(param, real(q_energies[:,0]) / delta, 'b', param, real(q_energies[:,1]) / delta, 'r')
xlabel('A or e')
ylabel('Quasienergy')
title('Floquet quasienergies')

figure(2)
plot(param, real(f_gnd_prob[:,0]), 'b', param, real(f_gnd_prob[:,1]), 'r')
xlabel('A or e')
ylabel('Occ. prob.')
title('Floquet modes excitation probability')

figure(3)
plot(param, real(wf_gnd_prob[:,0]), 'b', param, real(wf_gnd_prob[:,1]), 'r')
xlabel('A or e')
ylabel('Occ. prob.')
title('Floquet states excitation probability')
show()

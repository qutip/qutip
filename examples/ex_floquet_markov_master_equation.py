#
# Example: Find the floquet modes and quasi energies for a driven system and
# plot the floquet states/quasienergies for one period of the driving.
#
from qutip import *
from pylab import *
import time


def J_cb(omega):
    """ Noise spectral density """
    return omega
    
def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]
    w  = args[2]

    return H0 + cos(w * t) * H1

def qubit_integrate(delta, eps0, A, omega, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 =   A/2.0 * sz
        
    #H_args = (H0, H1, omega)
    H_args = {'w': omega}
    H = [H0, [H1, 'sin(w * t)']]
    
    # find the propagator for one driving period
    T = 2*pi / omega
       
    f_modes_0,f_energies = floquet_modes(H, T, H_args)

    c_op = sigmax()

    kmax = 1

    temp = 25e-3
    w_th = temp * (1.38e-23 / 6.626e-34) * 2 * pi * 1e-9
    
    Delta, X, Gamma, A = floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T, H_args, J_cb, w_th, kmax)

    k_idx = 0
    for k in range(-kmax,kmax+1, 1):
        print "X[",k,"] =\n", X[:,:,k_idx]
        k_idx += 1

    k_idx = 0
    for k in range(-kmax,kmax+1, 1):
        print "Delta[",k,"] =\n", Delta[:,:,k_idx]
        k_idx += 1

    k_idx = 0
    for k in range(-kmax,kmax+1, 1):
        print "Gamma[",k,"] =\n", Gamma[:,:,k_idx]
        k_idx += 1
        
    print "A =\n", A

    rho_ss = floquet_master_equation_steadystate(H0, A)
    
    
    R = floquet_master_equation_tensor(A)
    
    print "Floquet-Markov master equation tensor"
    
    print "R =\n", R
    
    print "Floquet-Markov master equation steady state =\n", rho_ss

    p_ex_0 = zeros(shape(tlist))
    p_ex_1 = zeros(shape(tlist))
    
    e_0 = zeros(shape(tlist))
    e_1 = zeros(shape(tlist))
        
    idx = 0
    for t in tlist:
        f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args) 

        p_ex_0[idx] = expect(sm.dag() * sm, f_modes_t[0])
        p_ex_1[idx] = expect(sm.dag() * sm, f_modes_t[1])

        #evals = hamiltonian_t(t, H_args).eigenenergies()
        evals = qobj_list_evaluate(H, t, H_args).eigenenergies()
        e_0[idx] = min(real(evals))
        e_1[idx] = max(real(evals))

        idx += 1
        
    return p_ex_0, p_ex_1, e_0, e_1, f_energies        
    
#
# set up the calculation: a strongly driven two-level system
# (repeated LZ transitions)
#
delta = 0.2 * 2 * pi  # qubit sigma_x coefficient
eps0  = 1.0 * 2 * pi  # qubit sigma_z coefficient
A     = 2.5 * 2 * pi  # sweep rate
psi0   = basis(2,0)   # initial state
omega  = 1.0 * 2 * pi # driving frequency
T      = (2*pi)/omega # driving period

tlist = linspace(0.0, 1 * T, 100)

start_time = time.time()
p_ex_0, p_ex_1, e_0, e_1, f_e = qubit_integrate(delta, eps0, A, omega, psi0, tlist)
print 'dynamics: time elapsed = ' + str(time.time() - start_time) 

#
# plot the results
#
figure(figsize=[8,10])
subplot(2,1,1)
plot(tlist, real(p_ex_0), 'b', tlist, real(p_ex_1), 'r')
xlabel('Time ($T$)')
ylabel('Excitation probabilities')
title('Floquet modes')
legend(("Floquet mode 1", "Floquet mode 2"))

subplot(2,1,2)
plot(tlist, real(e_0), 'c', tlist, real(e_1), 'm')
plot(tlist, ones(shape(tlist)) * f_e[0], 'b', tlist, ones(shape(tlist)) * f_e[1], 'r')
xlabel('Time ($T$)')
ylabel('Energy [GHz]')
title('Eigen- and quasi-energies')
legend(("Ground state", "Excited state", "Quasienergy 1", "Quasienergy 2"))
show()


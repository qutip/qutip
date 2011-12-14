#
# Example: Find the floquet modes and quasi energies for a driven system and
# evolve the wavefunction "stroboscopically", i.e., only by evaluating at 
# time mupliples of the driving period t = n * T for integer n.
#
# The system is a strongly driven two-level atom.
#
from qutip import *
from pylab import *
import time

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]
    w  = args[2]
    return H0 + sin(w * t) * H1

def qubit_integrate(delta, eps0, A, omega, psi0, tlist, T, option):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 =   A/2.0 * sz        
    H_args = (H0, H1, omega)


    if option == "floquet":

        # find the floquet modes for the time-dependent hamiltonian        
        f_modes_0,f_energies = floquet_modes(hamiltonian_t, T, H_args)

        # decompose the inital state in the floquet modes (=floquet states at t=0)
        f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)

                    
        # only evaluate the wavefunction at multiples of the driving period
        # i.e. a "stroboscopic" evolution
        N = max(tlist)/T
        p_ex = zeros(N)                    
        tlist = []
        
        # calculate the wavefunctions at times t=nT, for integer n, by using 
        # the floquet modes and quasienergies
        for n in arange(N):            
            psi_t = floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, n*T, hamiltonian_t, T, H_args)            
            p_ex[n] = expect(sm.dag() * sm, psi_t)
            tlist.append(n*T)   
    
    else:
    
        # for reference: evolve and calculate expectation using the full ode solver
        expt_list = odesolve(hamiltonian_t, psi0, tlist, [], [sm.dag() * sm], H_args)  
        p_ex = expt_list[0]
        
    return tlist, p_ex
    
#
# setup the calculation parameters and call.
#
delta = 0.2 * 2 * pi  # qubit sigma_x coefficient
eps0  = 0.1 * 2 * pi  # qubit sigma_z coefficient
A     = 1.0 * 2 * pi  # driving amplitude
psi0   = basis(2,0)   # initial state
omega  = 1.0 * 2 * pi # driving frequency

T      = (2*pi)/omega # driving period
tlist  = linspace(0.0, 25 * T, 500)

start_time = time.time()
tlist1, p_ex = qubit_integrate(delta, eps0, A, omega, psi0, tlist, T, "dynamics")
print 'dynamics: time elapsed = ' + str(time.time() - start_time) 

start_time = time.time()
tlist2, f_ex = qubit_integrate(delta, eps0, A, omega, psi0, tlist, T, "floquet")
print 'floquet: time elapsed = ' + str(time.time() - start_time) 


#
# plot the results
#
figure(figsize=[6,4])
plot(tlist1, real(p_ex),   'b')
plot(tlist1, real(1-p_ex), 'r')
plot(tlist2, real(f_ex),   'bo', linewidth=2.0)
plot(tlist2, real(1-f_ex), 'ro', linewidth=2.0)

xlabel('Time')
ylabel('Probability')
title('Stroboscopic time-evolution with')
legend(("ode $P_1$", "ode $P_0$", "Floquet $P_1$", "Floquet $P_0$"))
savefig('examples-floquet-evolution.png')
show()


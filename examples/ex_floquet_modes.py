#
# Example: Find the floquet modes and quasi energies for a driven system and
# plot the floquet states/quasienergies for one period of the driving.
#
from qutip import *
from pylab import *
import time

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]
    w  = args[2]

    return H0 + cos(w * t) * H1

def H1coeff_t(t, args):
    w  = args['w']
    return sin(w * t)           

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
    #H = [H0, [H1, H1coeff_t]]
    
    # find the propagator for one driving period
    T = 2*pi / omega
       
    f_modes_0,f_energies = floquet_modes(H, T, H_args)

    p_ex_0 = zeros(shape(tlist))
    p_ex_1 = zeros(shape(tlist))

    p_00 = zeros(shape(tlist), dtype=complex)
    p_01 = zeros(shape(tlist), dtype=complex)    
    p_10 = zeros(shape(tlist), dtype=complex)
    p_11 = zeros(shape(tlist), dtype=complex)    
    
    e_0 = zeros(shape(tlist))
    e_1 = zeros(shape(tlist))
         
    f_modes_table_t = floquet_modes_period_t(f_modes_0, f_energies, tlist, H, T, H_args) 

    for idx, t in enumerate(tlist):
        #f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args) 
        f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T) 

        p_ex_0[idx] = expect(sm.dag() * sm, f_modes_t[0])
        p_ex_1[idx] = expect(sm.dag() * sm, f_modes_t[1])

        p_00[idx] = f_modes_t[0].full()[0][0]
        p_01[idx] = f_modes_t[0].full()[1][0]
        p_10[idx] = f_modes_t[1].full()[0][0]
        p_11[idx] = f_modes_t[1].full()[1][0]

        #evals = hamiltonian_t(t, H_args).eigenenergies()
        evals = qobj_list_evaluate(H, t, H_args).eigenenergies()
        e_0[idx] = min(real(evals))
        e_1[idx] = max(real(evals))
        
    return p_ex_0, p_ex_1, e_0, e_1, f_energies, p_00, p_01, p_10, p_11
    
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

tlist = linspace(0.0, T, 101)

start_time = time.time()
p_ex_0, p_ex_1, e_0, e_1, f_e, p_00, p_01, p_10, p_11 = qubit_integrate(delta, eps0, A, omega, psi0, tlist)
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

#
# plot the results
#
figure(figsize=[8,12])
subplot(3,1,1)
plot(tlist, real(p_00), 'b', tlist, real(p_01), 'r')
plot(tlist, real(p_10), 'c', tlist, real(p_11), 'm')
xlabel('Time ($T$)')
ylabel('real')
title('Floquet modes')
legend(("FM1 - gnd", "FM1 - exc", "FM2 - gnd", "FM2 - exc"))

subplot(3,1,2)
plot(tlist, imag(p_00), 'b', tlist, imag(p_01), 'r')
plot(tlist, imag(p_10), 'c', tlist, imag(p_11), 'm')
xlabel('Time ($T$)')
ylabel('imag')
legend(("FM1 - gnd", "FM1 - exc", "FM2 - gnd", "FM2 - exc"))

subplot(3,1,3)
plot(tlist, abs(p_00), 'b', tlist, abs(p_01), 'r.')
plot(tlist, abs(p_10), 'c', tlist, abs(p_11), 'm.')
xlabel('Time ($T$)')
ylabel('abs')
legend(("FM1 - gnd", "FM1 - exc", "FM2 - gnd", "FM2 - exc"))



#
# finish by displaying graph windows
#
show()


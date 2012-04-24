#
# Example comparing the standard master equation with time-dependent hamiltonian
# with the floquet-markov master equation, in which the master equation is 
# derived in the floquet basis.
#
from qutip import *
from pylab import *
import time

gamma1 = 0.0015          # relaxation rate
gamma2 = 0.0           # dephasing  rate


def J_cb(omega):
    """ Noise spectral density """
    #print "evaluate J_cb for omega =", omega
    return 0.5 * gamma1 * omega/(2*pi)
    
def H1_coeff_t(t, args):
    return sin(args['w'] * t)

def qubit_integrate(delta, eps0, A, w, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A * sx

    args = {'w': w}
    H = [H0, [H1, 'sin(w * t)']]


    # --------------------------------------------------------------------------
    # 1) time-dependent hamiltonian
    # 
    #    
    c_op_list = [sqrt(gamma1) * sx, sqrt(gamma2) * sz]

    start_time = time.time()
    output = mesolve(H, psi0, tlist, c_op_list, [sm.dag() * sm], args=args)      
    expt_list1 = output.expect
    print 'Method 1: time elapsed = ' + str(time.time() - start_time)         

    # --------------------------------------------------------------------------
    # 2) Constant hamiltonian
    #
    H_rwa = - delta/2.0 * sx - A * sx / 2
    c_op_list = [sqrt(gamma1) * sx, sqrt(gamma2) * sz]
    
    start_time = time.time()
    output = mesolve(H_rwa, psi0, tlist, c_op_list, [sm.dag() * sm])  
    expt_list2 = output.expect
    print 'Method 2: time elapsed = ' + str(time.time() - start_time)           


    # --------------------------------------------------------------------------
    # 3) Floquet unitary evolution
    #
    import qutip.odeconfig
    qutip.odeconfig.tdfunc = None # better way of reseting this?!?
    
    start_time = time.time()
       
    T = 2*pi / w       
    f_modes_0,f_energies = floquet_modes(H, T, args)    
    
    # decompose the initial state vector in terms of the floquet modes (basis
    # transformation). used to calculate psi_t below.
    f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
    
    # --------------------------------------------------------------------------
    # 4) Floquet markov master equation dynamics
    #       
    kmax = 1
    temp = 25e-3
    w_th = temp * (1.38e-23 / 6.626e-34) * 2 * pi * 1e-9   
    
    f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, linspace(0, T, 500+1), H, T, args) 
    
    # calculate the rate-matrices for the floquet-markov master equation
    Delta, X, Gamma, Amat = floquet_master_equation_rates(f_modes_0, f_energies, sx, H, T, args, J_cb, w_th, kmax, f_modes_table_t)
   
    # the floquet-markov master equation tensor
    R = floquet_master_equation_tensor(Amat, f_energies)
    
    #expt_list4 = fmmesolve(R, f_modes_0, psi0, tlist, [sm.dag() * sm], opt=None) # note: in floquet basis...
    rho_list = fmmesolve(R, f_modes_0, psi0, tlist, [], opt=None) 

    expt_list3 = zeros(shape(expt_list2), dtype=complex)
    expt_list4 = zeros(shape(expt_list2), dtype=complex)
    for idx, t in enumerate(tlist):            
    
        # unitary floquet evolution
        psi_t = floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, args)            
        expt_list3[0][idx] = expect(sm.dag() * sm, psi_t) 
        
        # the rho_list returned by the floquet master equation is defined in the
        # floquet basis, so to transform it back to the computational basis
        # before we calculate expectation values.
        #f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, args)
        f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T) 
        expt_list4[0][idx] = expect((sm.dag() * sm), rho_list[idx].transform(f_modes_t, False)) 
    
    print 'Method 3+4: time elapsed = ' + str(time.time() - start_time)      

    # calculate the steadystate density matrix according to the floquet-markov
    # master equation
    rho_ss_fb = floquet_master_equation_steadystate(H0, Amat) # in floquet basis
    rho_ss_cb = rho_ss_fb.transform(f_modes_0, False)         # in computational basis
    expt_list5 = ones(shape(expt_list2), dtype=complex) * expect(sm.dag() * sm, rho_ss_cb)
    
    return expt_list1[0], expt_list2[0], expt_list3[0], expt_list4[0], expt_list5[0]
    
#
# set up the calculation
#
delta = 0.0 * 2 * pi   # qubit sigma_x coefficient
eps0  = 1.0 * 2 * pi   # qubit sigma_z coefficient
A     = 0.05 * 2 * pi   # driving amplitude (reduce to make the RWA more accurate)
w     = 1.0 * 2 * pi   # driving frequency
psi0 = (0.3*basis(2,0)+0.7*basis(2,1)).unit()      # initial state

tlist = linspace(0, 30.0, 500)

p_ex1, p_ex2, p_ex3, p_ex4, p_ex5 = qubit_integrate(delta, eps0, A, w, gamma1, gamma2, psi0, tlist)

figure()
plot(tlist, real(p_ex1), 'b', tlist, real(p_ex2), 'g-') # lindblad
plot(tlist, real(p_ex3), 'r', tlist, real(p_ex4), 'm-', tlist, real(p_ex5), 'c-') # floquet markov
xlabel('Time')
ylabel('Occupation probability')
title('Comparison between time-dependent master equations')
legend(("TD Hamiltonian, Std ME", "RWA Hamiltonian, Std ME", "Unitary Floquet evol.", "Floquet-Markov ME", "F-M ME steady state"))
show()



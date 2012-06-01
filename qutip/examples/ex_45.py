#
# Calculate the dynamics of a driven two-level system with according to the 
# Floquet-Markov master equation. For compari
#
from qutip import *
from pylab import *
import qutip.odeconfig

gamma1 = 0.05 # relaxation rate
gamma2 = 0.0  # dephasing  rate

def J_cb(omega):
    """ Noise spectral density """
    return 0.5 * gamma1 * omega/(2*pi)

def run():

    delta = 0.0 * 2 * pi # qubit sigma_x coefficient
    eps0  = 1.0 * 2 * pi # qubit sigma_z coefficient
    A     = 0.1 * 2 * pi # driving amplitude
    w     = 1.0 * 2 * pi # driving frequency
    T     = 2*pi / w     # driving period
    psi0  = basis(2,0)   # initial state
    tlist = linspace(0, 25.0, 250)

    # Hamiltonian: list-string format
    args = {'w': w}
    H0 = - delta/2.0 * sigmax() - eps0/2.0 * sigmaz()
    H1 = - A * sigmax()
    H = [H0, [H1, 'sin(w * t)']]

    # --------------------------------------------------------------------------
    # Standard lindblad master equation with time-dependent hamiltonian
    # 
    c_op_list = [sqrt(gamma1) * sigmax(), sqrt(gamma2) * sigmaz()]
    p_ex_me = mesolve(H, psi0, tlist, c_op_list, [num(2)], args=args).expect[0]
     
    # --------------------------------------------------------------------------
    # Floquet markov master equation dynamics
    #       
    qutip.odeconfig.tdfunc = None # reset td func flag
    
    # find initial floquet modes and quasienergies
    f_modes_0,f_energies = floquet_modes(H, T, args, False)    
           
    # precalculate floquet modes for the first driving period
    f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, linspace(0, T, 500+1), H, T, args) 
    
    # solve the floquet-markov master equation
    rho_list = fmmesolve(H, psi0, tlist, sigmax(), [], [J_cb], T, args).states

    # calculate expectation values in the computational basis
    p_ex_fmme = zeros(shape(p_ex_me), dtype=complex)
    for idx, t in enumerate(tlist):
        f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T) 
        p_ex_fmme[idx] = expect(num(2), rho_list[idx].transform(f_modes_t, False)) 
        
    # plot the results
    figure()
    plot(tlist, real(p_ex_me), 'b')  # standard lindblad with time-dependence
    plot(tlist, real(p_ex_fmme), 'm-') # floquet markov
    xlabel('Time')
    ylabel('Occupation probability')
    legend(("Standard Lindblad ME", "Floquet-Markov ME"))
    show()

if __name__=='__main__':
    run()

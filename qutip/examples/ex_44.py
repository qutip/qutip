#
# Calculate the quasienergies for a driven two-level system as a function of
# driving amplitude. See Creffield et al., Phys. Rev. B 67, 165301 (2003).
#

from qutip import *
from pylab import *

def run():

    delta   = 1.0 * 2 * pi  # bare qubit sigma_z coefficient
    epsilon = 0.0 * 2 * pi  # bare qubit sigma_x coefficient
    omega   = 8.0 * 2 * pi  # driving frequency
    T       = (2*pi)/omega  # driving period

    E_vec = linspace(0.0, 12.0, 100) * omega

    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    q_energies = zeros((len(E_vec), 2))

    H0 = delta/2.0 * sz - epsilon/2.0 * sx
    for idx, A in enumerate(E_vec):
        H1 = A/2.0 * sx
       
        # H = H0 + H1 * sin(w * t) in the 'list-string' format
        args = {'w': omega}
        H = [H0, [H1, 'cos(w * t)']]
            
        # find the floquet modes
        f_modes,f_energies = floquet_modes(H, T, args)

        q_energies[idx,:] = f_energies
        
    # plot the results
    plot(E_vec/omega, real(q_energies[:,0]) / delta, 'b', E_vec/omega, real(q_energies[:,1]) / delta, 'r')
    xlabel(r'$E/\omega$')
    ylabel(r'Quasienergy / $\Delta$')
    title(r'Floquet quasienergies')
    text(4, 0.4, r'$H = \frac{\Delta}{2}\sigma_z + \frac{E}{2}\cos(\omega t)\sigma_x$', fontsize=20)
    show()
    
if __name__=='__main__':
    run()


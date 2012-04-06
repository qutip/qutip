#
# Steady state and photon occupation number for a sideband-cooled
# nanomechanical resonator, as a function of the ambient temperature.
#
from qutip.Qobj import *
from qutip.tensor import *
from qutip.ptrace import *
from qutip.operators import *
from qutip.expect import *
from qutip.wigner import *
from qutip.steady import *
from qutip.odesolve import mesolve

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# constants
hbar = 6.626e-34
kB   = 1.38e-23

#
# calculate the steadystate average photon count in the two resonators as a 
# function of Temperature, with and without sideband cooling driving
#
def compute(T_vec, N_r, N_m, w_r, w_m, g, w_d, A, kappa_r, kappa_m):
    
    # pre-calculate operators
    a = tensor(destroy(N_r), qeye(N_m)) # for high-freq. mode
    b = tensor(qeye(N_r), destroy(N_m)) # for mechanical mode
    
    # Hamiltonian with driving, in the corresponding RWA
    H = (w_r - w_d) * a.dag() * a + w_m * b.dag() * b + \
        g * (a.dag() * a) * (b + b.dag()) + 0.5 * A * (a + a.dag())

     
    photon_count = zeros((len(T_vec), 4))
       
    for idx, T in enumerate(T_vec):       
       
        # tempeature in frequency units [2*pi GHz]
        w_th = kB * (T * 1e-3) / hbar * 1e-9

        # collapse operators
        c_ops = []

        # collapse operators for high-frequency resonator
        n_r_th = 1.0 / (exp(w_r/w_th) - 1.0)
        rate = kappa_r * (1 + n_r_th)
        if rate > 0.0: c_ops.append(sqrt(rate) * a)
        rate = kappa_r * n_r_th
        if rate > 0.0: c_ops.append(sqrt(rate) * a.dag())

        # collapse operators for mechanical mode
        n_m_th = 1.0 / (exp(w_m/w_th) - 1.0)
        rate = kappa_m * (1 + n_m_th)
        if rate > 0.0: c_ops.append(sqrt(rate) * b)
        rate = kappa_m * n_m_th
        if rate > 0.0: c_ops.append(sqrt(rate) * b.dag())

        # find the steady state
        rho_ss = steadystate(H, c_ops)
        
        # calculate the photon numbers 
        photon_count[idx,0] = expect(rho_ss, a.dag() * a)
        photon_count[idx,1] = n_r_th
        photon_count[idx,2] = expect(rho_ss, b.dag() * b)
        photon_count[idx,3] = n_m_th

    return photon_count
    
def run():

    # Configure parameters
    N_r = 4         # number of fock states in high-frequency resonator
    N_m = 10        # number of fock states in mechanical resonator

    w_r = 2*pi*10.0  # high-freq. resonator frequency [2*pi GHz] 
    w_m = 2*pi*0.25  # mechanical resonator frequency [2*pi GHz]
    
    g  = 2*pi*0.01   # coupling strength

    w_d = w_r-w_m    # driving frequency, selected to match resonance condition
    A   = 2*pi*0.05   # driving amplitude in frequency units

    kappa_r = 0.001  # dissipation rate for high-frequency resoantor
    kappa_m = 0.001  # dissipation rate for mechanical resonator
    
    T_vec = linspace(0.0, 100.0, 25.0) # Temperature [mK]

    # find the steady state occupation numbers
    photon_count = compute(T_vec, N_r, N_m, w_r, w_m, g, w_d, A, kappa_r, kappa_m)

    # plot the results
    figure()
    
    plot(T_vec, photon_count[:,0], 'b')
    plot(T_vec, photon_count[:,1], 'b:')
    plot(T_vec, photon_count[:,2], 'r')
    plot(T_vec, photon_count[:,3], 'r:')
     
    xlabel(r'Temperature [mK]',fontsize=14)
    ylabel(r'Occupation number',fontsize=14)
    title("Average photon occupation number\n" +
          "in a sideband-cooled mechanical resonator")
    
    legend(("High-freq. resonator (%.2f GHz)" % (w_r/(2*pi)), 
            "High-freq. resonator, no cooling",
            "Mech. resonator (%.2f GHz)" % (w_m/(2*pi)), 
            "Mech. resonator, no cooling"), loc=2)       
    show()

if __name__=='__main__':
    run()


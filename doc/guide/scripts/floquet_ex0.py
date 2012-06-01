from qutip import *

delta = 0.2 * 2*pi; eps0  = 0.0 * 2*pi
omega = 1.0 * 2*pi; A_vec = linspace(0, 10, 100) * omega;
T      = (2*pi)/omega 
tlist  = linspace(0.0, 10 * T, 101)
psi0   = basis(2,0) 

q_energies = zeros((len(A_vec), 2))

H0 = delta/2.0 * sigmaz() - eps0/2.0 * sigmax()
args = omega
for idx, A in enumerate(A_vec):
    H1 = A/2.0 * sigmax()
    H = [H0, [H1, lambda t, w: sin(w*t)]]    
    f_modes,f_energies = floquet_modes(H, T, args, True)
    q_energies[idx,:] = f_energies
    
# plot the results
from pylab import *
plot(A_vec/omega, real(q_energies[:,0]) / delta, 'b', \
     A_vec/omega, real(q_energies[:,1]) / delta, 'r')
xlabel(r'$A/\omega$')
ylabel(r'Quasienergy / $\Delta$')
title(r'Floquet quasienergies')
show()

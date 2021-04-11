import numpy as np
from matplotlib import pyplot
import qutip

delta = 0.2 * 2*np.pi
eps0  = 0.0 * 2*np.pi
omega = 1.0 * 2*np.pi
A_vec = np.linspace(0, 10, 100) * omega
T      = 2*np.pi/omega
tlist  = np.linspace(0.0, 10 * T, 101)
psi0   = qutip.basis(2, 0)

q_energies = np.zeros((len(A_vec), 2))

H0 = delta/2.0 * qutip.sigmaz() - eps0/2.0 * qutip.sigmax()
args = omega
for idx, A in enumerate(A_vec):
    H1 = A/2.0 * qutip.sigmax()
    H = [H0, [H1, lambda t, w: np.sin(w*t)]]
    f_modes,f_energies = qutip.floquet_modes(H, T, args, True)
    q_energies[idx,:] = f_energies

# plot the results
pyplot.plot(
    A_vec/omega, np.real(q_energies[:, 0]) / delta, 'b',
    A_vec/omega, np.real(q_energies[:, 1]) / delta, 'r',
)
pyplot.xlabel(r'$A/\omega$')
pyplot.ylabel(r'Quasienergy / $\Delta$')
pyplot.title(r'Floquet quasienergies')
pyplot.show()

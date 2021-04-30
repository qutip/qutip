import numpy as np
from matplotlib import pyplot

import qutip

delta = 0.2 * 2*np.pi
eps0  = 1.0 * 2*np.pi
A     = 0.5 * 2*np.pi
omega = 1.0 * 2*np.pi
T      = (2*np.pi)/omega
tlist  = np.linspace(0.0, 10 * T, 101)
psi0   = qutip.basis(2, 0)

H0 = - delta/2.0 * qutip.sigmax() - eps0/2.0 * qutip.sigmaz()
H1 = A/2.0 * qutip.sigmaz()
args = {'w': omega}
H = [H0, [H1, lambda t,args: np.sin(args['w'] * t)]]

# find the floquet modes for the time-dependent hamiltonian
f_modes_0,f_energies = qutip.floquet_modes(H, T, args)

# decompose the inital state in the floquet modes
f_coeff = qutip.floquet_state_decomposition(f_modes_0, f_energies, psi0)

# calculate the wavefunctions using the from the floquet modes
p_ex = np.zeros(len(tlist))
for n, t in enumerate(tlist):
    psi_t = qutip.floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, args)
    p_ex[n] = qutip.expect(qutip.num(2), psi_t)

# For reference: calculate the same thing with mesolve
p_ex_ref = qutip.mesolve(H, psi0, tlist, [], [qutip.num(2)], args).expect[0]

# plot the results
pyplot.plot(tlist, np.real(p_ex),     'ro', tlist, 1-np.real(p_ex),     'bo')
pyplot.plot(tlist, np.real(p_ex_ref), 'r',  tlist, 1-np.real(p_ex_ref), 'b')
pyplot.xlabel('Time')
pyplot.ylabel('Occupation probability')
pyplot.legend(("Floquet $P_1$", "Floquet $P_0$", "Lindblad $P_1$", "Lindblad $P_0$"))
pyplot.show()

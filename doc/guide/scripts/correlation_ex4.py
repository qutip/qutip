import numpy as np
import matplotlib.pyplot as plt
import qutip

N = 25
taus = np.linspace(0, 25.0, 200)
a = qutip.destroy(N)
H = 2 * np.pi * a.dag() * a

kappa = 0.25
n_th = 2.0  # bath temperature in terms of excitation number
c_ops = [np.sqrt(kappa * (1 + n_th)) * a, np.sqrt(kappa * n_th) * a.dag()]

states = [
    {'state': qutip.coherent_dm(N, np.sqrt(2)), 'label': "coherent state"},
    {'state': qutip.thermal_dm(N, 2), 'label': "thermal state"},
    {'state': qutip.fock_dm(N, 2), 'label': "Fock state"},
]

fig, ax = plt.subplots(1, 1)

for state in states:
    rho0 = state['state']

    # first calculate the occupation number as a function of time
    n = qutip.mesolve(H, rho0, taus, c_ops, e_ops=[a.dag() * a]).expect[0]

    # calculate the correlation function G2 and normalize with n(0)n(t) to
    # obtain g2
    G2 = qutip.correlation_3op_1t(H, rho0, taus, c_ops, a.dag(), a.dag()*a, a)
    g2 = np.array(G2) / (n[0] * np.array(n))

    ax.plot(taus, np.real(g2), label=state['label'], lw=2)

ax.legend(loc=0)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$g^{(2)}(\tau)$')
plt.show()

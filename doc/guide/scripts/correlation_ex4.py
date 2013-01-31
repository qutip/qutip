from qutip import *

N = 25
taulist = linspace(0, 25.0, 200)
a = destroy(N)
H = 2 * pi * a.dag() * a

import pylab as plt
fig, ax = plt.subplots(1, 1)

kappa = 0.25
n_th = 2.00  # bath temperature in terms of excitation number
c_ops = [sqrt(kappa*(1+n_th)) * a, sqrt(kappa*n_th) * a.dag()]

states = [{'state': coherent_dm(N, sqrt(2.0)), 'label': "coherent state"}, 
          {'state': thermal_dm(N, 2.0), 'label': "thermal state"},
          {'state': fock_dm(N, 2), 'label': "Fock state"}]

for state in states:

    rho0 = state['state']

    # first calculate the occupation number as a function of time
    n = mesolve(H, rho0, taulist, c_ops, [a.dag() * a]).expect[0]

    # calculate the correlation function G2 and normalize with n^2 to obtain g2
    G2 = correlation_ss_gtt(H, taulist, c_ops, a.dag(), a.dag(), a, a, rho0=rho0)
    g2 = G2 / n**2

    ax.plot(taulist, real(g2), label=state['label'])

ax.legend(loc=0)
ax.set_xlabel(r'$\tau$');
ax.set_ylabel(r'$g^{(2)}(\tau)$');
plt.show()

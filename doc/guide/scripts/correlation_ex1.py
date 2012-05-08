from qutip import *

N = 15
taulist = linspace(0,10.0,200)
a = destroy(N)
H = 2*pi*a.dag()*a

# collapse operator
G1 = 0.75
n_th = 2.00  # bath temperature in terms of excitation number
c_ops = [sqrt(G1*(1+n_th)) * a, sqrt(G1*n_th) * a.dag()]

# start with a coherent state
rho_t1 = coherent_dm(N, 2.0)

# first calculate the occupation number as a function of time
n = mesolve(H, rho_t1, taulist, c_ops, [a.dag() * a]).expect[0]

# calculate the correlation function G1 and normalize with n to obtain g1
G1 = correlation(H, rho_t1, None, taulist, c_ops, a.dag(), a)
g1 = G1 / sqrt(n[0] * n)

from pylab import *
plot(taulist, g1, 'b')
plot(taulist, n, 'r')
title('Decay of a coherent state to an incoherent (thermal) state')
xlabel(r'$\tau$')
legend((r'First-order coherence function $g^{(1)}(\tau)$', 
        r'occupation number $n(\tau)$'))
show()

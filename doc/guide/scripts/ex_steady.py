import numpy as np
import matplotlib.pyplot as plt

import qutip

# Define paramters
N = 20  # number of basis states to consider
a = qutip.destroy(N)
H = a.dag() * a
psi0 = qutip.basis(N, 10)  # initial state
kappa = 0.1  # coupling to oscillator

# collapse operators
c_op_list = []
n_th_a = 2  # temperature with average of 2 excitations
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * a)  # decay operators
rate = kappa * n_th_a
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * a.dag())  # excitation operators

# find steady-state solution
final_state = qutip.steadystate(H, c_op_list)
# find expectation value for particle number in steady state
fexpt = qutip.expect(a.dag() * a, final_state)

tlist = np.linspace(0, 50, 100)
# monte-carlo
mcdata = qutip.mcsolve(H, psi0, tlist, c_op_list, e_ops=[a.dag() * a], ntraj=100)
# master eq.
medata = qutip.mesolve(H, psi0, tlist, c_op_list, e_ops=[a.dag() * a])

plt.plot(tlist, mcdata.expect[0], tlist, medata.expect[0], lw=2)
# plot steady-state expt. value as horizontal line (should be = 2)
plt.axhline(y=fexpt, color='r', lw=1.5)
plt.ylim([0, 10])
plt.xlabel('Time', fontsize=14)
plt.ylabel('Number of excitations', fontsize=14)
plt.legend(('Monte-Carlo', 'Master Equation', 'Steady State'))
plt.title(
    r'Decay of Fock state $\left|10\rangle\right.$'
    r' in a thermal environment with $\langle n\rangle=2$'
)
plt.show()

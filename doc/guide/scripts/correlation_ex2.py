import numpy as np
import matplotlib.pyplot as plt
import qutip

times = np.linspace(0, 10.0, 200)
a = qutip.destroy(10)
x = a.dag() + a
H = a.dag() * a
alpha = 2.5
rho0 = qutip.coherent_dm(10, alpha)
corr = qutip.correlation_2op_2t(H, rho0, times, times, [np.sqrt(0.25) * a], x, x)

plt.pcolor(np.real(corr))
plt.xlabel(r'Time $t_2$')
plt.ylabel(r'Time $t_1$')
plt.title(r'Correlation $\left<x(t)x(0)\right>$')
plt.show()

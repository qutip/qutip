import numpy as np
import matplotlib.pyplot as plt
import qutip

times = np.linspace(0, 10, 200)
a = qutip.destroy(10)
x = a.dag() + a
H = a.dag() * a

corr1 = qutip.correlation_2op_1t(H, None, times, [np.sqrt(0.5) * a], x, x)
corr2 = qutip.correlation_2op_1t(H, None, times, [np.sqrt(1.0) * a], x, x)
corr3 = qutip.correlation_2op_1t(H, None, times, [np.sqrt(2.0) * a], x, x)

plt.plot(times, np.real(corr1), times, np.real(corr2), times, np.real(corr3))
plt.xlabel(r'Time $t$')
plt.ylabel(r'Correlation $\left<x(t)x(0)\right>$')
plt.show()

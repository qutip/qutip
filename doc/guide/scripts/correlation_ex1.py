import numpy as np
import pylab as plt
from qutip import *

times = np.linspace(0,10.0,200)
a = destroy(10)
x = a.dag() + a
H = a.dag() * a

corr1 = correlation_2op_1t(H, None, times, [np.sqrt(0.5) * a], x, x)
corr2 = correlation_2op_1t(H, None, times, [np.sqrt(1.0) * a], x, x)
corr3 = correlation_2op_1t(H, None, times, [np.sqrt(2.0) * a], x, x)

plt.plot(times, np.real(corr1), times, np.real(corr2), times, np.real(corr3))
plt.xlabel(r'Time $t$')
plt.ylabel(r'Correlation $\left<x(t)x(0)\right>$')
plt.show()

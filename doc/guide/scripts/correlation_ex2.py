from qutip import *
from scipy import *

times = linspace(0, 10.0, 200)
a = destroy(10)
x = a.dag() + a
H = a.dag() * a
alpha = 2.5
rho0 = coherent_dm(10, alpha)
corr = correlation(H, rho0, times, times, [sqrt(0.25) * a], x, x)

from pylab import *
pcolor(corr)
xlabel(r'Time $t_2$')
ylabel(r'Time $t_1$')
title(r'Correlation $\left<x(t)x(0)\right>$')
show()

from qutip import *

tlist = linspace(0,10.0,200)
a  = destroy(10)
x  = a.dag() + a
H  = a.dag()*a
alpha = 2.5
corr = correlation(H, coherent_dm(10, alpha), tlist, tlist, [sqrt(0.25)*a], x, x)

from pylab import *
pcolor(corr)
xlabel(r'Time $t_2$')
ylabel(r'Time $t_1$')
title(r'Correlation $\left<x(t)x(0)\right>$')
show()


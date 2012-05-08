from qutip import *

tlist = linspace(0,10.0,200)
a  = destroy(10)
x  = a.dag() + a
H = a.dag()*a

corr1 = correlation_ss(H, tlist, [sqrt(0.5)*a], x, x)
corr2 = correlation_ss(H, tlist, [sqrt(1.0)*a], x, x)
corr3 = correlation_ss(H, tlist, [sqrt(2.0)*a], x, x)

from pylab import *
plot(tlist, real(corr1), tlist, real(corr2), tlist, real(corr3))
xlabel(r'Time $t$')
ylabel(r'Correlation $\left<x(t)x(0)\right>$')
show()

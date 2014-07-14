from qutip import *
from scipy import *

times = linspace(0,10.0,200)
a = destroy(10)
x = a.dag() + a
H = a.dag() * a

corr1 = correlation_ss(H, times, [sqrt(0.5) * a], x, x)
corr2 = correlation_ss(H, times, [sqrt(1.0) * a], x, x)
corr3 = correlation_ss(H, times, [sqrt(2.0) * a], x, x)

from pylab import *
plot(times, real(corr1), times, real(corr2), times, real(corr3))
xlabel(r'Time $t$')
ylabel(r'Correlation $\left<x(t)x(0)\right>$')
show()

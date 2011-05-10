from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
N = 20;
g = 2;
psi = basis(N,10) + basis(N,5)
xvec = arange(-5.,5.,.1)
yvec = xvec
X,Y = meshgrid(xvec, yvec)
Q = qfunc(psi,xvec,xvec,g);
fig1 = figure()
ax = Axes3D(fig1)
ax.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, alpha=0.7)
show()


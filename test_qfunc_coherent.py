from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *

N = 6;
g = 2;
xvec = arange(-40.,40.)*5./40
X,Y = meshgrid(xvec, xvec)
rho = coherent_dm(N, 1)
print "rho =", rho
Q = qfunc(rho,xvec,xvec,g);
fig1 = plt.figure()
ax = Axes3D(fig1)
ax.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, alpha=0.7)
xlabel('re alpha')
ylabel('im alpha')
plt.show()


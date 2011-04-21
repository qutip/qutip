#
#
#
#
from scipy import *
from scipy.linalg import *
import scipy.sparse as sp
import scipy.linalg as la


from qutip import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *

N = 20;
g = 2;

psi = basis(N,10) + basis(N,5)

print "psi = ", psi

xvec = arange(-40.,40.)*5./40
yvec = xvec
X,Y = meshgrid(xvec, yvec)

#W=wigner(psi,xvec,xvec)
#print "W = ", W

Q = qfunc(psi,xvec,xvec,g);
print "Q = ", Q

fig1 = plt.figure()
ax = Axes3D(fig1)
ax.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, alpha=0.7)
#ax.contour(X, Y, Q, levels=15, zdir='z', offset=-0.6)
#ax.set_zlim3d(-0.4,0.2)
#ax.set_zlim3d(0.,0.)

plt.show()


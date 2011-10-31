from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
N = 20;
psi=(coherent(N,-2-2j)+coherent(N,2+2j)).unit()
xvec = linspace(-5.,5.,200)
yvec = xvec
X,Y = meshgrid(xvec, yvec)
W = wigner(psi,xvec,xvec);
#
# First plot the wigner function:
#
fig = plt.figure(figsize=(6, 4))
ax = Axes3D(fig,azim=-107,elev=49)
surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0,linewidth=0.05)
fig2.colorbar(surf, shrink=0.65, aspect=20)
savefig('examples-schcatdist-w.png')
close(fig)
#
# Now plot the Q-function:
#
Q = qfunc(psi,xvec,xvec)
fig = plt.figure(figsize=(6, 4))
ax = Axes3D(fig,azim=-107,elev=49)
surf=ax.plot_surface(X, Y, Q, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0,linewidth=0.05)
fig2.colorbar(surf, shrink=0.65, aspect=20)
savefig('examples-schcatdist-q.png')
close(fig)

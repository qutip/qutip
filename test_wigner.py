from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
N = 20;
psi=(coherent(N,-2-2j)+coherent(N,2+2j)).unit()
#psi = ket2dm(basis(N,0))
xvec = linspace(-5.,5.,200)
yvec = xvec
X,Y = meshgrid(xvec, yvec)
W = wigner(psi,xvec,xvec);
fig2 = plt.figure(figsize=(9, 6))
ax = Axes3D(fig2,azim=-107,elev=49)
surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0,linewidth=0.05)
fig2.colorbar(surf, shrink=0.65, aspect=20)
savefig("test.png")
#show()


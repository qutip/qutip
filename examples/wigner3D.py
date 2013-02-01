from qutip import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
x = 1.0 / sqrt(2) * (basis(10, 4) + basis(10, 2))
xvec = arange(-5, 5, 10.0 / 100)
yvec = xvec
W = wigner(x, xvec, yvec)
from qutip.graph import wigner_cmap
cmap = wigner_cmap(W)
X, Y = meshgrid(xvec, yvec)
contourf(X, Y, W, 50, cmap=cmap)
colorbar()
show()
fig = figure()
ax = Axes3D(fig, azim=-30, elev=73)
ax.plot_surface(X, Y, W, cmap=cmap, rstride=1, cstride=1, alpha=1, linewidth=0)
ax.set_zlim3d(-0.25, 0.25)
for a in ax.w_zaxis.get_ticklines() + ax.w_zaxis.get_ticklabels():
    a.set_visible(False)
nrm = mpl.colors.Normalize(W.min(), W.max())
cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.02)
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=nrm)
cb1.set_label('Pseudoprobability')
show()

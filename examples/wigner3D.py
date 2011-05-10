from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,mpl
from pylab import *
x=1.0/sqrt(2)*(basis(10,4) + basis(10,2))
xvec=arange(-5,5,10.0/100)
yvec=xvec
W=wigner(x,xvec,yvec)
X,Y = meshgrid(xvec, yvec)
contourf(X, Y, W,50)
colorbar()
show()
fig =figure()
ax = Axes3D(fig,azim=-30,elev=73)
ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.copper, alpha=1,linewidth=0)
ax.set_zlim3d(-0.25,0.25)
for a in ax.w_zaxis.get_ticklines()+ax.w_zaxis.get_ticklabels():
    a.set_visible(False)
nrm=mpl.colors.Normalize(W.min(),W.max())
cax,kw=mpl.colorbar.make_axes(ax,shrink=.66,pad=.02)
cb1=mpl.colorbar.ColorbarBase(cax,cmap=cm.copper,norm=nrm)
cb1.set_label('Pseudoprobability')
show()




#
#
#
import time
import os
from qutip import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *

x=basis(100,4) + basis(100,2)
xvec=arange(-50., 50.)*4./30
yvec=xvec
tic=time.clock()
W=wigner(x,xvec,yvec)
toc=time.clock()
time=toc-tic
print 'calculation time = ',time,' secs'
X,Y = meshgrid(xvec, yvec)
pcolor(X, Y, W)
colorbar()
show()
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, W, rstride=2, cstride=2, cmap=cm.jet, alpha=.9)
ax.contour(X,Y,W,levels=15,zdir='z', offset=-0.6)
ax.set_zlim3d(-0.6,0.6)
#savefig('wigner-test.pdf',format='pdf')
plt.show()




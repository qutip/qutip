from qutip import *
from pylab import *
from scipy import random
from matplotlib import pyplot, mpl,cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

Q=Qobj(random.random((6,6))) #density matrix with random entries

num_elem=prod(Q.shape) #num. of elements to plot
xpos,ypos=meshgrid(range(Q.shape[0]),range(Q.shape[1]))
xpos=xpos.T.flatten()-0.5 #center bars on integer value of x-axis
ypos=ypos.T.flatten()-0.5 #center bars on integer value of y-axis
zpos = zeros(num_elem) #all bars start at z=0
dx =0.75*ones(num_elem) #width of bars in x-direction
dy = dx.copy() #width of bars in y-direction (same as x-dir here)
dz = Q.full().flatten() #height of bars from density matrix elements (should use 'real()' if complex)

#nrm=mpl.colors.Normalize(0,max(dz)) #<-- normalize colors to max. data
nrm=mpl.colors.Normalize(0,1) #<-- normalize colors to 1
colors=cm.jet(nrm(dz)) #list of colors for each bar

#plot figure
fig = plt.figure(figsize=(6,4))
ax = Axes3D(fig,azim=-40,elev=70)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
ax.axes.w_xaxis.set_major_locator(IndexLocator(1,-0.5)) #set x-ticks to integers
ax.axes.w_yaxis.set_major_locator(IndexLocator(1,-0.5)) #set y-ticks to integers
ax.axes.w_zaxis.set_major_locator(IndexLocator(1,0)) #set z-ticks to integers
ax.set_zlim3d([0,1.1])

cax,kw=mpl.colorbar.make_axes(ax,shrink=.75,pad=.02) #add colorbar with normalized range
cb1=mpl.colorbar.ColorbarBase(cax,cmap=cm.jet,norm=nrm)

savefig('examples-3d-histogram.png')


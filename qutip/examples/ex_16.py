#
# Creation and manipulation of a Bell state with
# 3D histogram plot output.
#

from qutip.Qobj import *
from qutip.states import *
from qutip.operators import *
from qutip.ptrace import *
from qutip.tensor import *
from pylab import *
from matplotlib import pyplot, mpl,cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def qubit_hist(Q,xlabels,ylabels,title):
    # Plots density matrix 3D histogram from Qobj
    # xlabels and ylabels are list of strings for axes tick labels
    num_elem=prod(Q.shape) #num. of elements to plot
    xpos,ypos=meshgrid(range(Q.shape[0]),range(Q.shape[1]))
    xpos=xpos.T.flatten()-0.5 #center bars on integer value of x-axis
    ypos=ypos.T.flatten()-0.5 #center bars on integer value of y-axis
    zpos = zeros(num_elem) #all bars start at z=0
    dx =0.8*ones(num_elem) #width of bars in x-direction
    dy = dx.copy() #width of bars in y-direction (same as x-dir here)
    dz = real(Q.full().flatten()) #height of bars from density matrix elements.
    
    #generate list of colors for probability data
    nrm=mpl.colors.Normalize(min(dz)-0.1,max(dz)+0.1) # add +-0.1 in case all elements
                                                      # are the same (colorbar will fail).
    colors=cm.jet(nrm(dz))

    #plot figure
    fig = plt.figure()
    ax = Axes3D(fig,azim=-47,elev=85)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    #set x-axis tick labels and label font size
    ax.axes.w_xaxis.set_major_locator(IndexLocator(1,-0.5)) #set x-ticks to integers
    ax.set_xticklabels(xlabels) 
    ax.tick_params(axis='x', labelsize=18)

    #set y-axis tick labels and label font size
    ax.axes.w_yaxis.set_major_locator(IndexLocator(1,-0.5)) #set y-ticks to integers
    ax.set_yticklabels(ylabels) 
    ax.tick_params(axis='y', labelsize=18)

    #remove z-axis tick labels by moving them outside the plot range
    ax.axes.w_zaxis.set_major_locator(IndexLocator(2,2)) #set z-ticks to integers
    #set the plot range in the z-direction to fit data 
    ax.set_zlim3d([min(dz)-0.1,max(dz)+0.1])
    plt.title(title)
    #add colorbar with color range normalized to data
    cax,kw=mpl.colorbar.make_axes(ax,shrink=.75,pad=.02)
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=cm.jet,norm=nrm)
    cb1.set_label("Probability",fontsize=14)
    show()
    



def run():
    #create Bell state
    up = basis(2,0)
    dn = basis(2,1)
    bell = (tensor([up,up])+tensor([dn,dn])).unit()
    rho_bell=ket2dm(bell)

    x_bell_labels=['$\left|00\\rangle\\right.$','$\left|01\\rangle\\right.$',
                    '$\left|10\\rangle\\right.$','$\left|11\\rangle\\right.$']
    y_bell_labels=x_bell_labels
    title='Bell state density matrix'
    #plot Bell state density matrix
    qubit_hist(rho_bell,x_bell_labels,y_bell_labels,title)

    #trace over qubit 0
    bell_trace_1=ptrace(rho_bell,1)
    xlabels=['$\left|0\\rangle\\right.$','$\left|1\\rangle\\right.$']
    ylabels=xlabels
    title='Partial trace over qubit 0 in Bell state'
    #plot remaining qubit density matrix
    qubit_hist(bell_trace_1,xlabels,ylabels,title)

    #create projection operator
    left = (up + dn).unit()
    Omegaleft = tensor(qeye(2),left*left.dag())
    after = Omegaleft*bell
    after=ket2dm(after/after.norm())
    title="Bell state after (up + dn) projection operator on qubit 1"
    #plot density matrix after projection
    qubit_hist(after,x_bell_labels,y_bell_labels,title)

    #plot partial trace of state 1
    after_trace=ptrace(after,1)
    title="Completely mixed state after partial trace over qubit 0"
    qubit_hist(after_trace,xlabels,ylabels,title)


if __name__=='__main__':
    run()
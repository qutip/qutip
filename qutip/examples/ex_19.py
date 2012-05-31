#
# Plots the entangled superposition
# 3-qubit GHZ eigenstate |up,up,up>+|dn,dn,dn>
#
# From the xGHZ qotoolbox example by Sze M. Tan
#
from qutip import *
from pylab import *
from matplotlib import pyplot, mpl,cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def run():
    #create spin operators for the three qubits.
    sx1=tensor(sigmax(),qeye(2),qeye(2))
    sy1=tensor(sigmay(),qeye(2),qeye(2))

    sx2=tensor(qeye(2),sigmax(),qeye(2))
    sy2=tensor(qeye(2),sigmay(),qeye(2))

    sx3=tensor(qeye(2),qeye(2),sigmax())
    sy3=tensor(qeye(2),qeye(2),sigmay())
    
    #Calculate products
    op1=sx1*sy2*sy3
    op2=sy1*sx2*sy3
    op3=sy1*sy2*sx3
    opghz=sx1*sx2*sx3

    # Find simultaneous eigenkets of op1,op2,op3 and opghz
    evalues,states=simdiag([op1,op2,op3,opghz])
    
    #plot the density matrix for the entangled |up,up,up>+|dn,dn,dn>
    #state using same qubit histrogram as the 'Bell State' example.
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
        nrm=mpl.colors.Normalize(min(dz),max(dz))
        colors=cm.jet(nrm(dz))

        #plot figure
        fig = plt.figure()
        ax = Axes3D(fig,azim=-15,elev=75)
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

        #set x-axis tick labels and label font size
        ax.axes.w_xaxis.set_major_locator(IndexLocator(1,-0.5)) #set x-ticks to integers
        ax.set_xticklabels(xlabels) 
        ax.tick_params(axis='x', labelsize=14)

        #set y-axis tick labels and label font size
        ax.axes.w_yaxis.set_major_locator(IndexLocator(1,-0.5)) #set y-ticks to integers
        ax.set_yticklabels(ylabels) 
        ax.tick_params(axis='y', labelsize=14)

        #remove z-axis tick labels by moving them outside the plot range
        ax.axes.w_zaxis.set_major_locator(IndexLocator(2,2)) #set z-ticks to integers
        #set the plot range in the z-direction to fit data 
        ax.set_zlim3d([min(dz)-0.1,max(dz)+0.1])
        plt.title(title)
        #add colorbar with color range normalized to data
        cax,kw=mpl.colorbar.make_axes(ax,shrink=.75,pad=.02)
        cb1=mpl.colorbar.ColorbarBase(cax,cmap=cm.jet,norm=nrm)
        show()
        
    
    #convert last eigenstate to density matrix
    rho0=ket2dm(states[-1])
    #create labels for density matrix plot
    upupup="$|\\uparrow,\\uparrow,\\uparrow\\rangle$"
    dndndn="$|\\downarrow,\\downarrow,\\downarrow\\rangle$"
    title="3-Qubit GHZ state: $\\frac{1}{\\sqrt{2}}$"+upupup+"+"+dndndn
    xlabels=[""]*8
    xlabels[0]=upupup  #set first xaxes label
    xlabels[-1]=dndndn #set last xaxes label
    ylabels=[""]*8
    ylabels[-1]=upupup #set last yaxes label
    #generate plot with labels
    qubit_hist(rho0,xlabels,ylabels,title)
    
    
    
if __name__=='__main__':
    run()

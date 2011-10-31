.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

Trilinear Hamiltonian: Deviation from a thermal spectrum
---------------------------------------------------------

Adapted from: P. D. Nation and M. P. Blencowe, "The Trilinear Hamiltonian: a zero-dimensional model of Hawking radiation from a quantized source", NJP *12* 095013 (2010)

The parametric amplifier is a common example of a linear amplifier.  It is well-known that the parametric amplifier produces a thermal state, when starting from vacuum, in both the signal or idler mode, when the other mode is traced over.  The key approximation in the parametric amplifier is the assumption that the pump mode be modeled as a classical system, defined by a c-number amplitude and phase.  Relaxing this condition leads to the trilinear Hamiltonian, where the pump is now a quantized degree of freedom.  As the example below shows, the signal or idler mode begins to deviate from a thermal distribution as the pump mode transfers energy.::
    
    from qutip import *
    from pylab import *
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    #number of states for each mode
    N0=6
    N1=6
    N2=6

    #define operators
    a0=tensor(destroy(N0),qeye(N1),qeye(N2))
    a1=tensor(qeye(N0),destroy(N1),qeye(N2))
    a2=tensor(qeye(N0),qeye(N1),destroy(N2))

    #number operators for each mode
    num0=a0.dag()*a0
    num1=a1.dag()*a1
    num2=a2.dag()*a2

    #initial state: coherent mode #0 & vacuum for modes #1 & #2
    alpha=sqrt(2) #initial coherent state with expt. value of two particles
    psi0=tensor(coherent(N0,alpha),basis(N1,0),basis(N2,0))

    #trilinear Hamiltonian
    H=1.0j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

    #run Monte-Carlo
    tlist=linspace(0,2.5,50)
    states=mcsolve(H,psi0,tlist,1,[],[])

    mode1=[ptrace(k,1) for k in states]                  #trace out all but mode #1
    diags1=[real(k.diag()) for k in mode1]               #get diagonal elements (number state probabilities)
    num1=[expect(num1,k) for k in states]                #expectation values for particles in mode #1
    thermal=[thermal_dm(N1,k).diag() for k in num1] #number state probabilities of a thermal state defined by num1

    #set plotting parameters
    params = {'axes.labelsize': 14,'text.fontsize': 14,'legend.fontsize': 12,'xtick.labelsize': 14,'ytick.labelsize': 14}
    rcParams.update(params)

    colors=['m', 'g','orange','b', 'y','pink'] #colors used for data sets
    x=range(N1) #defines x-axis of plot

    fig = plt.figure()
    ax = Axes3D(fig)
    for j in range(5): #add bar plot for each probability distribution
        ax.bar(x, diags1[10*j], zs=tlist[10*j], zdir='y',color=colors[j],linewidth=1.0,alpha=0.6,align='center') #actual dist.
        ax.plot(x,thermal[10*j],zs=tlist[10*j],zdir='y',color='r',linewidth=3,alpha=1) #plots red line for expected thermal dist.
    ax.set_zlabel(r'Probability')
    ax.set_xlabel(r'Number State')
    ax.set_ylabel(r'Time')
    ax.set_zlim3d(0,1)
    show()

.. figure:: http://qutip.googlecode.com/svn/wiki/images/trilinear_deviations.png
    :align: center
    :target: http://qutip.googlecode.com/svn/wiki/images/trilinear_deviations.png
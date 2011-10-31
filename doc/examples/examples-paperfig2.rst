.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

Figure 2 from the QuTiP manuscript.
-------------------------------------------------------------------------------

Here we calculate the ground state occupation probability for a coupled cavity-qubit system in the ultra-strong coupling regime as a function of the coupling strength g.  In set figure shows the Wigner function for the cavity mode at the largest coupling strength :math:`g=2.5`, which is well approximated by Eq.4 from the paper.::
    
    from qutip import *
    ## set up the calculation ## 
    wc = 1.0 * 2 * pi # cavity frequency
    wa = 1.0 * 2 * pi # atom frequency
    N = 20            # number of cavity states
    g = linspace(0, 2.5, 50)*2*pi # coupling strength vector
    ## create operators ## 
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    nc = a.dag() * a
    na = sm.dag() * sm
    ## initialize output arrays ##
    na_expt = zeros(len(g))
    nc_expt = zeros(len(g))
    ## run calculation ## 
    for k in range(len(g)):
        ## recalculate the hamiltonian for each value of g ## 
        H = wc*nc+wa*na+g[k]*(a.dag()+a)*(sm+sm.dag())
        ## find the groundstate ## 
        ekets, evals = H.eigenstates()
        psi_gnd = ekets[0]
        ## expectation values ## 
        na_expt[k] = expect(na, psi_gnd) # qubit occupation
        nc_expt[k] = expect(nc, psi_gnd) # cavity occupation 
    ## Calculate Wigner function for coupling g=2.5 ## 
    rho_cavity = ptrace(psi_gnd,0) # trace out qubit
    xvec = linspace(-7.5,7.5,200)
    ## Wigner function ## 
    W = wigner(rho_cavity, xvec, xvec)


    # ------------------------------------------------------------------------------
    # Plot the results (omitted from the code listing in the appendix in the paper)
    #
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Times New Roman'
    rcParams['legend.fontsize'] = 16
    from qutip import *
    from pylab import *
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot, mpl,cm
    from matplotlib.ticker import MaxNLocator

    #
    # plot the cavity and atom occupation numbers as a function of 
    #
    fig=figure(1)
    ax = fig.add_subplot(111)
    ax2=ax.twinx()
    ax2.plot(g/(2*pi), na_expt, lw=2)
    ax2.plot(g/(2*pi), nc_expt, 'r--', lw=2)
    ax.set_xlabel(r'Coupling strength $g/\omega_{0}$',fontsize=18)
    ax2.set_ylabel(r'Occupation number',fontsize=18)
    for a in ax.yaxis.get_ticklines()+ax.yaxis.get_ticklabels():
        a.set_visible(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax2.yaxis.get_ticklabels():
        tick.set_fontsize(16)

    #
    # calculate wigner function of cavity mode at final coupling strength g=2.5.
    #
    fig = plt.figure(figsize=(7,5))
    X,Y = meshgrid(xvec, xvec)

    #
    # plot the cavity wigner function.
    #
    ax = Axes3D(fig, azim=-61, elev=43)
    surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet, alpha=1.0, linewidth=0.0, vmax=0.15, vmin=-0.05)
    ax.set_xlim3d(-7.5, 7.5)
    ax.set_xlabel(r'position',fontsize=16)
    ax.set_ylim3d(-7.5, 7.5)
    ax.set_ylabel(r'momentum',fontsize=16)
    ax.w_xaxis.set_major_locator(MaxNLocator(5))
    ax.w_yaxis.set_major_locator(MaxNLocator(5))
    ax.w_zaxis.set_major_locator(MaxNLocator(5))
    for tick in ax.w_xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.w_yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for a in ax.axes.w_zaxis.get_ticklines()+ax.axes.w_zaxis.get_ticklabels():
        a.set_visible(False)
    cax,kw=mpl.colorbar.make_axes(ax,shrink=.66,pad=-.075)
    nrm=mpl.colors.Normalize(W.min(),W.max())
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=cm.jet,norm=nrm)
    cb1.set_label('Probability',fontsize=18)
    cb1.set_ticks(linspace(round(W.min(),1),round(W.max(),1),6))
    for t in cb1.ax.get_yticklabels():
         t.set_fontsize(16)
    show()


.. figure:: http://qutip.googlecode.com/svn/wiki/images/paper_fig2_1.png 
    :align: center
    :target: http://qutip.googlecode.com/svn/wiki/images/paper_fig2_1.png 

.. figure:: http://qutip.googlecode.com/svn/wiki/images/paper_fig2_2.png 
    :align: center
    :target: http://qutip.googlecode.com/svn/wiki/images/paper_fig2_2.png

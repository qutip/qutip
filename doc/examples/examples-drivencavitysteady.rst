.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

Finds steady-state solution to driven system of cavity + qubit
---------------------------------------------------------------
  
Here we will find the steady-state solution to the problem of a driven-cavity coupled to a two-level (qubit) system for various driving frequencies.  This example also demonstrates the use of the :func:`qutip.parfor` function.  Adapted from the qotoolbox example 'probss' by Sze. M. Tan::
    
    from qutip import * #loads qutip
    from pylab import * #loads matplotlib

    #function for solving the steady-state dynamics
    def probss(E,kappa,gamma,g,wc,w0,wl,N):
        ida=qeye(N)            #identity operator for cavity
        idatom=qeye(2)     #identity operator for qubit
        a=tensor(destroy(N),idatom) #destruction operator for cavity excitations for cavity+qubit system
        sm=tensor(ida,sigmam())      #destruction operator for qubit excitations for cavity+qubit system

        #Hamiltonian
        H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)

        #Collapse operators
        C1=sqrt(2*kappa)*a
        C2=sqrt(gamma)*sm
        C1dC1=C1.dag() * C1
        C2dC2=C2.dag() * C2

        #Liouvillian
        L = liouvillian(H, [C1, C2])

        #find steady state
        rhoss=steady(L)

        #calculate expectation values
        count1=expect(C1dC1,rhoss)
        count2=expect(C2dC2,rhoss)
        infield=expect(a,rhoss)
        return count1,count2,infield

    #-------------------------------------------------------------------------------
    # setup the calculation
    #-------------------------------------------------------------------------------
    kappa=2            #mirror coupling
    gamma=0.2      #spontaneous emission rate
    g=1                    #atom-cavity coupling
    wc=0                  #cavity frequency
    w0=0                 #atomic frequency
    N=5                    #size of Hilbert space for the cavity (zero to N-1 photons) 
    E=0.5                 #amplitude of driving field
    nloop=101
    wlist=linspace(-5,5,nloop) #array of driving field frequency's 

    #define single-variable function of driving frequency for use in parfor
    def func(wl):#function of wl only
        count1,count2,infield=probss(E,kappa,gamma,g,wc,w0,wl,N)
        return count1,count2,infield

    #run simulation by looping over wl array in parallel using parfor
    [count1,count2,infield] = parfor(func,wlist)

    fig=figure()
    plot(wlist,count1,wlist,count2)
    xlabel('Detuning')
    ylabel('Count rates')
    show()
    close(fig)

    fig=figure()
    plot(wlist,180.0*angle(infield)/pi)
    xlabel('Detuning')
    ylabel('Intracavity phase shift')
    show()

.. figure:: http://qutip.googlecode.com/svn/wiki/images/ss-countrates.png
    :align: center
    :target: http://qutip.googlecode.com/svn/wiki/images/ss-countrates.png
    
.. figure:: http://qutip.googlecode.com/svn/wiki/images/ss-phaseshift.png
    :align: center
    :target: http://qutip.googlecode.com/svn/wiki/images/ss-phaseshift.png



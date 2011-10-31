.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

Example on how to calculate two-time correlation functions in QuTiP.
--------------------------------------------------------------------
  
In the following example we calculate the <x(t)x(0)> correlation function for a cavity, with and without coupling to a two-level atom.::

    
    #
    # Example: Calculate the <x(t)x(0)> correlation function for a cavity that is
    # coupled to a qubit.
    #
    from qutip import *
    from pylab import *

    def calc_correlation(N, wc, wa, g, kappa, gamma, tlist):

        # Hamiltonian
        a  = tensor(destroy(N), qeye(2))
        sm = tensor(qeye(N), destroy(2))
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())

        # collapse operators
        c_op_list = []

        n_th_a = 0.0
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * a)

        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * a.dag())

        rate = gamma
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm)

        # correlation function operator
        x = a + a.dag()

        # calculate the correlation function
        return correlation_ss_ode(H, tlist, c_op_list, x, x)

    #
    # setup the calcualtion
    #
    N = 10              # number of cavity fock states
    wc = 1.0 * 2 * pi   # cavity frequency
    wa = 1.0 * 2 * pi   # atom frequency
    g  = 0.5 * 2 * pi   # coupling strength
    kappa = 0.25        # cavity dissipation rate
    gamma = 0.0         # atom dissipation rate

    tlist = linspace(0, 5, 200)

    corr1 = calc_correlation(N, wc, wa,   g, kappa, gamma, tlist)
    corr2 = calc_correlation(N, wc, wa, 0.0, kappa, gamma, tlist)

    figure(1)
    plot(tlist,real(corr1), tlist, real(corr2))
    xlabel('Time')
    ylabel('Correlation <x(t)x(0)>')
    legend(("g = 0.5", "g = 0.0"))
    show()

.. figure:: http://qutip.googlecode.com/svn/wiki/images/examples-correlation-function.png
    :align: center
    :target: http://qutip.googlecode.com/svn/wiki/images/examples-correlation-function.png
    



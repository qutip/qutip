.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _tensor:

*********************************************
Visualization of quantum states and processes
*********************************************

.. ipython::
   :suppress:

   In [1]: from qutip import *


.. _tensor-products:

Visualization is often an important complement to a simulation of a quantum
mechanical system. The first method of visualization that comes to mind might be
to plot the expectation values of a few selected operators. But on top of that,
it can often be instructive to visualize for example the state vectors or
density matices that describe the state of the system, or how the state is
transformed as a function of time (see process tomography below). In this 
section we demonstrate how QuTiP and matplotlib can be used to perform a few
types of  visualizations that often can provide additional understanding of
quantum system.


Probability distribution functions
==================================

In quantum mechanics probability distributions plays an important role, and as
in statistics, the expectation values computed from a probability distribution
does not reveal the full story. For example, consider an quantum harmonic
oscillator mode with Hamiltonian :math:`H = \hbar\omega a^\dagger a`, which is 
in a state described by its density matrix :math:`\rho`, and which on average
is occupied by two photons, :math:`\mathrm{Tr}[\rho a^\dagger a] = 2`. Given
this information we cannot say whether the oscillator is in a Fock state, 
a thermal state, a coherent state, etc. By visualizing the photon distribution
in the Fock state basis important clues about the underlaying state can be
obtained.

One convenient way to visualize a probability distribution is to use histograms.
Consider the following histogram visualization of a few possible oscillator 
states with on average occupation of two photons.

.. ipython::

    In [1]: N = 20

    In [1]: fig, axes = subplots(1, 3, figsize=(12,3))

    In [1]: # coherent state

    In [1]: rho = coherent_dm(N, 2)

    In [1]: bar0 = axes[0].bar(range(0, N), real(rho.diag()))

    In [1]: lbl0 = axes[0].set_title("Coherent state")

    In [1]: # thermal state

    In [1]: rho = thermal_dm(N, 2)

    In [1]: bar1 = axes[1].bar(range(0, N), real(rho.diag()))

    In [1]: lbl1 = axes[1].set_title("Thermal state")

    In [1]: # Fock state

    In [1]: rho = fock_dm(N, 2)

    In [1]: bar2 = axes[2].bar(range(0, N), real(rho.diag()))

    In [1]: lbl2 = axes[2].set_title("Fock state")

    In [1]: lim2 = axes[2].set_xlim([0, N])

	@savefig visualization-test.png width=8.0in align=center
    In [1]: show()


All these states correspond to an average of two photons, but by visualizing
the photon distribution in Fock basis one can gain additional understanding
about the nature and properties of the state. 


Quasi-probability distributions
===============================



Visualizing operators
=====================




Quantum process tomography
==========================



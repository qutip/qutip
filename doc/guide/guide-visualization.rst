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


Fock-basis probability distribution
===================================

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
Consider the following histogram visualization of the number-basis probability
distribution, which can be obtained from the diagonal of the density matrix, 
for a few possible oscillator states with on average occupation of two photons.

First we generate the density matrices for the coherent, thermal and fock states.

.. ipython::

    In [1]: N = 20

    In [1]: rho_coherent = coherent_dm(N, sqrt(2))

    In [1]: rho_thermal = thermal_dm(N, 2)

    In [1]: rho_fock = fock_dm(N, 2)


Next, we plot histograms of the diagonals of the density matrices:

.. ipython::

    In [1]: fig, axes = subplots(1, 3, figsize=(12,3))

    In [1]: bar0 = axes[0].bar(range(0, N), real(rho_coherent.diag()))

    In [1]: lbl0 = axes[0].set_title("Coherent state")

    In [1]: bar1 = axes[1].bar(range(0, N), real(rho_thermal.diag()))

    In [1]: lbl1 = axes[1].set_title("Thermal state")

    In [1]: bar2 = axes[2].bar(range(0, N), real(rho_fock.diag()))

    In [1]: lbl2 = axes[2].set_title("Fock state")

    In [1]: lim2 = axes[2].set_xlim([0, N])

	@savefig visualization-distribution.png width=8.0in align=center
    In [1]: show()


All these states correspond to an average of two photons, but by visualizing
the photon distribution in Fock basis the differences between these states are
easily appreciated. 

Quasi-probability distributions
===============================

The probability distribution in the number (Fock) basis only describes the
occupation probabilities for a discrete set of states. A more complete
phase-space probability-distribution-like function for harmonic modes are 
the Wigner and Husumi Q-functions, which are full descriptions of the 
quantum state (equivalent to the density matrix). These are called
quasi-distribution functions because unlike real probability distribution
functions they can for example be negative. In addition to being more complete descriptions
of a state (compared to only the occupation probabilities plotted above),
these distributions are also great for demonstrating if a quantum state is
quantum mechanical, since for example a negative Wigner function
is a definite indicator that a state is distinctly nonclassical.


Wigner function
---------------

In QuTiP, the Wigner function for a harmonic mode can be calculated with the
function :func:`qutip.wigner.wigner`. It takes a ket or a density matrix as 
input, together with arrays that define the ranges of the phase space
coordinates (in x and y plane). In the following example the Wigner functions
are calculated and plotted for the same three states as in the previous section.

.. ipython::

    In [1]: xvec = linspace(-5,5,200)

    In [1]: W_coherent = wigner(rho_coherent, xvec, xvec)

    In [1]: W_thermal = wigner(rho_thermal, xvec, xvec)

    In [1]: W_fock = wigner(rho_fock, xvec, xvec)

    In [1]: # plot the results

    In [1]: fig, axes = subplots(1, 3, figsize=(12,3))

    In [1]: cont0 = axes[0].contourf(xvec, xvec, W_coherent, 100)

    In [1]: lbl0 = axes[0].set_title("Coherent state")

    In [1]: cont1 = axes[1].contourf(xvec, xvec, W_thermal, 100)

    In [1]: lbl1 = axes[1].set_title("Thermal state")

    In [1]: cont0 = axes[2].contourf(xvec, xvec, W_fock, 100)

    In [1]: lbl2 = axes[2].set_title("Fock state")

	@savefig visualization-wigner.png width=8.0in align=center
    In [1]: show()

Husumi Q-function
-----------------

.. ipython::

    In [1]: N = 20

    In [1]: xvec = linspace(-5,5,200)

    In [1]: fig, axes = subplots(1, 3, figsize=(12,3))

    In [1]: # coherent state

    In [1]: rho = coherent_dm(N, sqrt(2))

    In [1]: W = qfunc(rho, xvec, xvec)

    In [1]: cont0 = axes[0].contourf(xvec, xvec, W, 100)

    In [1]: lbl0 = axes[0].set_title("Coherent state")

    In [1]: # thermal state

    In [1]: rho = thermal_dm(N, 2)

    In [1]: W = qfunc(rho, xvec, xvec)

    In [1]: cont1 = axes[1].contourf(xvec, xvec, W, 100)

    In [1]: lbl1 = axes[1].set_title("Thermal state")

    In [1]: # Fock state

    In [1]: rho = fock_dm(N, 2)

    In [1]: W = qfunc(rho, xvec, xvec)

    In [1]: cont0 = axes[2].contourf(xvec, xvec, W, 100)

    In [1]: lbl2 = axes[2].set_title("Fock state")

	@savefig visualization-q-func.png width=8.0in align=center
    In [1]: show()

Visualizing operators
=====================




Quantum process tomography
==========================



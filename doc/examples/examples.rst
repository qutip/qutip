.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

QuTiP Examples
**************

Running QuTiP demos
===================

QuTip includes 20 built in demos from the examples below that demonstrate the usage of the built in functions for solving a variety of problems.  To run the demos, load the QuTiP package::

    >>> from qutip import *
    
and run the demos function::

    >>> demos()

This will generate the examples GUI, or a command line list of demos, depending on the availability of the graphics libraries:

.. figure:: http://qutip.googlecode.com/svn/wiki/images/demos.png
    :align: center

If you do not have any graphics libraries installed, or they are disabled, then the demos function *must* be run from the terminal.


Basic examples
==============

.. toctree::
   :maxdepth: 2
  
   examples-wigner.rst
   examples-squeezed.rst
   examples-schcatdist.rst
   examples-energy-spectrum.rst

Basics
------

*[ExamplesEnergySpectrum Energy spectrum of a coupled three qubit system.]*

*[ExamplesUltrastrong Cavity occupation number and Wigner function in the ultrastrong coupling regime.]* *(QuTiP paper figure #2)*

*[ExamplesEntropy Von-Neumann entropy of a binary mixture of |up> and |down> states]*

*[ExamplesBar3D Visualizing a density matrix as a 3D histogram]*

*[ExamplesAngular Plotting an angular wave function and direction eigen-ket]*

*[ExamplesGHZ Simultaneous diagonalization of operators using simdiag to generate GHZ states]*

== Master equation ==
*[ExamplesJaynesCummingsModel Vacuum Rabi oscillations in the Jaynes-Cummings model with dissipation.]*

*[ExamplesJCWignerEvolve Wigner distributions from the master equation evolution of the Jaynes-Cummings model]*

*[ExamplesSingleAtomLasing Single-atom lasing in a Jaynes-Cumming-like system, with an additional incoherent pump]*

*[ExamplesSinglePhotonSource Single photon source based on a three level atom strongly coupled to a cavity]*

*[ExamplesHeisenbergSpinChain Heisenberg spin-chain example]*

*[ExamplesBlochQubitDecay Decay of a qubit state coupled to a zero-temp. bath shown on a Bloch sphere.]*

*[ExamplesTimeDependence Rabi oscillation in a two-level system subject to a time-dependent classical driving field.]*

*[ExamplesLandauZener Landau-Zener transitions in a quantum two-level system]* *(QuTiP paper figures #10 & 11)*

*[ExamplesFidelity  Measuring the distance between density matrices via the fidelity]*

Monte-Carlo
===========

.. toctree::
   :maxdepth: 2
   
   examples-expectmonte.rst
   examples-trilinearmonte.rst
   examples-thermalmonte.rst
   examples-parampmonte.rst
   examples-collapsetimesmonte.rst


Steady state calculations
=========================

.. toctree::
   :maxdepth: 2
  
   examples-corrfunc.rst
   examples-drivencavitysteady.rst
   examples-spectrumsteady.rst

QuTiP Manuscript Figure Scripts
===============================

Additional examples corresponding to figures 2, 4, 6, 7, 8, 10, and 11 from the [http://arxiv.org/abs/1110.0573 QuTiP paper] may be found in the [http://code.google.com/p/qutip/downloads/list downloads section].

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
   examples-ultrastrong.rst
   examples-entropy.rst
   examples-3d-histogram.rst
   examples-angular.rst
   examples-GHZ.rst      

Master equation
===============

.. toctree::
   :maxdepth: 2
  
   examples-jc-model.rst
   examples-jc-model-wigner.rst
   examples-lasing.rst
   examples-single-photon-source.rst

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

.. toctree::
   :maxdepth: 2
   
   examples-paperfig2.rst
   examples-paperfig4.rst
   examples-paperfig6.rst
   examples-paperfig7.rst
   examples-paperfig8.rst
   examples-paperfig10_11.rst

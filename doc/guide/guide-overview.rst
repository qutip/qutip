.. QuTiP 
   Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson

.. _overview:

******************
Guide Overview
******************

The goal of this guide is to introduce you to the basic structures and functions that make up QuTiP. This guide is divided up into several sections, each highlighting a specific set of functionalities. In combination with the examples that can be found on the project web page `http://qutip.org/tutorials.html <http://qutip.org/tutorials.html>`_, this guide should provide a more or less complete overview. In addition, the :ref:`apidoc` for each function is located at the end of this guide.


.. _overview-org:

Organization
=============

QuTiP is designed to be a general framework for solving quantum mechanics problems such as systems composed of few-level quantum systems and harmonic oscillators. To this end, QuTiP is built from a large (and ever growing) library of functions and classes; from :func:`qutip.states.basis` to :func:`qutip.wigner`.  The general organization of QuTiP, highlighting the important API available to the user, is shown in the :ref:`figure-qutip_org`


.. _figure-qutip_org:

.. figure:: qutip_tree.pdf
   :align: center
   :figwidth: 100%
   
   Tree-diagram of the 306 user accessible functions and classes in QuTiP 3.2.

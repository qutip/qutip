.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

Introduction
*************

.. figure:: figures/wide_logo.png
   :align: center

QuTiP is open-source software for simulating the dynamics of 
open quantum systems.  The QuTiP library depends on the 
excellent Numpy and Scipy numerical packages. In addition, 
graphical output is provided by Matplotlib.  QuTiP aims
to provide user-friendly and efficient numerical simulations
of a wide variety of Hamiltonian's, including those with 
arbitrary time-dependence, commonly found in a wide range of 
physics applications. QuTiP is freely available for use and/or 
modification on all major platforms. Being free of any licensing 
fees, QuTiP is ideal for exploring quantum mechanics and 
dynamics in the classroom.

The focus of QuTiP is on systems where the Hamiltonian is composed of harmonic oscillator modes, and multi-level (typically two-levels) atoms, where the Hilbert space may be constructed using the Fock basis.  This large class of Hamiltonians appears in a wide variety of physical problems, including those in quantum optics, trapped ions, superconducting circuit devices, and mechanical systems in the quantum regime.  Although similar software options are already available, QuTiP differs from these alternatives by combining the following benefits:

* QuTiP relies on completely open-source software.  You are free to modify and use it as you wish with no licensing fees.

* QuTiP is based on the Python scripting language, providing easy to read, fast code generation without the need to compile after modification.

* The numerics underlying QuTiP are time-tested algorithms that run at C-code speeds, thanks to the `Numpy <http://numpy.scipy.org/>`_ and `Scipy <http://www.scipy.org/ scipy>`_ libraries, and are based on many of the same algorithms used in propriety software.

* QuTiP allows for solving the dynamics of Hamiltonians with arbitrary time-dependence (does not yet include time-dependent collapse operators).

* Takes advantage of multiple processors found in essentially all modern computers.

* QuTiP was designed from the start to require a minimal learning curve for those users who have experience using the popular quantum optics toolbox by Sze M. Tan. 

* Includes the ability to create high-quality plots, and animations, using the excellent `Matplotlib <http://matplotlib.sourceforge.net/>`_ package.

Organization
=============

.. figure:: figures/qutip_org.png
   :align: center

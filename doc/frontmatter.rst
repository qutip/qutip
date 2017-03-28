.. QuTiP 
   Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

.. _frontmatter:

*************
Frontmatter
*************

.. _about-docs:

About This Documentation
==========================

This document contains a user guide and automatically generated API documentation for QuTiP. A PDF version of this text is available at the `documentation page <http://www.qutip.org/documentation.html>`_. 

**For more information see the** `QuTiP project web page`_.

.. _QuTiP project web page: http://www.qutip.org


:Author: P.D. Nation

:Author: Alexander Pitchford

:Author: Arne Grimsmo

:Author: J.R. Johansson

:Author: Chris Grenade


:version: 4.1
:status: Released (March 10, 2017)
:copyright: This documentation is licensed under the Creative Commons Attribution 3.0 Unported License.

.. _citing-qutip:

Citing This Project
==========================
    
If you find this project useful, then please cite:

.. centered:: J. R. Johansson, P.D. Nation, and F. Nori, "QuTiP 2: A Python framework for the dynamics of open quantum systems", Comp. Phys. Comm. **184**, 1234 (2013).

or

.. centered:: J. R. Johansson, P.D. Nation, and F. Nori, "QuTiP: An open-source Python framework for the dynamics of open quantum systems", Comp. Phys. Comm. **183**, 1760 (2012).


which may also be download from http://arxiv.org/abs/1211.6518 or http://arxiv.org/abs/1110.0573, respectively.

.. _funding-qutip:

Funding
=======

The development of QuTiP has been partially supported by the Japanese Society for the Promotion of Science Foreign Postdoctoral Fellowship Program under grants P11202 (PDN) and P11501 (JRJ).  Additional funding comes from RIKEN, Kakenhi grant Nos. 2301202 (PDN), 2302501 (JRJ), and Korea University. 

.. _image-jsps:

.. figure:: figures/jsps.jpg
   :width: 4in
   :figclass: align-center

.. _image-riken:

.. figure:: figures/riken-logo.png
	:width: 1.5in
	:figclass: align-center

|

.. _image-korea:

.. figure:: figures/korea-logo.png
	:width: 3in
	:figclass: align-center


.. _about:

About QuTiP
===========

Every quantum system encountered in the real world is an open quantum system. For although much care is taken experimentally to eliminate the unwanted influence of external interactions, there remains, if ever so slight, a coupling between the system of interest and the external world. In addition, any measurement performed on the system necessarily involves coupling to the measuring device, therefore introducing an additional source of external influence. Consequently, developing the necessary tools, both theoretical and numerical, to account for the interactions between a system and its environment is an essential step in understanding the dynamics of quantum systems.

In general, for all but the most basic of Hamiltonians, an analytical description of the system dynamics is not possible, and one must resort to numerical simulations of the equations of motion. In absence of a quantum computer, these simulations must be carried out using classical computing techniques, where the exponentially increasing dimensionality of the underlying Hilbert space severely limits the size of system that can be efficiently simulated. However, in many fields such as quantum optics, trapped ions, superconducting circuit devices, and most recently nanomechanical systems, it is possible to design systems using a small number of effective oscillator and spin components, excited by a limited number of quanta, that are amenable to classical simulation in a truncated Hilbert space.

The Quantum Toolbox in Python, or QuTiP, is a fully open-source implementation of a framework written in the Python programming language designed for simulating the open quantum dynamics for systems such as those listed above. This framework distinguishes itself from the other available software solutions in providing the following advantages:

* QuTiP relies entirely on open-source software.  You are free to modify and use it as you wish with no licensing fees or limitations.

* QuTiP is based on the Python scripting language, providing easy to read, fast code generation without the need to compile after modification.

* The numerics underlying QuTiP are time-tested algorithms that run at C-code speeds, thanks to the `Numpy <http://numpy.scipy.org/>`_, `Scipy <http://www.scipy.org/scipylib>`_, and `Cython <http://cython.org>`_ libraries, and are based on many of the same algorithms used in propriety software.

* QuTiP allows for solving the dynamics of Hamiltonians with (almost) arbitrary time-dependence, including collapse operators.

* Time-dependent problems can be automatically compiled into C-code at run-time for increased performance.

* Takes advantage of the multiple processing cores found in essentially all modern computers.

* QuTiP was designed from the start to require a minimal learning curve for those users who have experience using the popular quantum optics toolbox by Sze M. Tan. 

* Includes the ability to create high-quality plots, and animations, using the excellent `Matplotlib <http://matplotlib.sourceforge.net/>`_ package.


For detailed information about new features of each release of QuTiP, see the :ref:`changelog`.


Contributing to QuTiP
=====================
We welcome anyone who is interested in helping us make QuTiP the best package for simulating quantum systems. Anyone who contributes will be duly recognized.  Even small contributions are noted. See :ref:`developers-contributors` for a list of people who have helped in one way or another. If you are interested, please drop us a line at the `QuTiP discussion group webpage`_. 


.. _QuTiP discussion group webpage: http://groups.google.com/group/qutip.


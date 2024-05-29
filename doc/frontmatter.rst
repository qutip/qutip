.. _frontmatter:

*************
Frontmatter
*************

.. _about-docs:

About This Documentation
==========================

This document contains a user guide and automatically generated API documentation for QuTiP. A PDF version of this text is available at the `documentation page <https://qutip.org/documentation.html>`_.

**For more information see the** `QuTiP project web page`_.

.. _QuTiP project web page: https://qutip.org/


:Author: J.R. Johansson

:Author: P.D. Nation

:Author: Alexander Pitchford

:Author: Arne Grimsmo

:Author: Chris Grenade

:Author: Nathan Shammah

:Author: Shahnawaz Ahmed

:Author: Neill Lambert

:Author: Eric Giguere

:Author: Boxi Li

:Author: Jake Lishman

:Author: Simon Cross

:Author: Asier Galicia

:Author: Paul Menczel

:Author: Patrick Hopf

:release: |release|

:copyright:
   The text of this documentation is licensed under the Creative Commons Attribution 3.0 Unported License.
   All contained code samples, and the source code of QuTiP, are licensed under the 3-clause BSD licence.
   Full details of the copyright notices can be found on the `Copyright and Licensing <copyright>`_ page of this documentation.

.. _citing-qutip:

Citing This Project
==========================

If you find this project useful, then please cite:

.. centered:: J. R. Johansson, P.D. Nation, and F. Nori, "QuTiP 2: A Python framework for the dynamics of open quantum systems", Comp. Phys. Comm. **184**, 1234 (2013).

or

.. centered:: J. R. Johansson, P.D. Nation, and F. Nori, "QuTiP: An open-source Python framework for the dynamics of open quantum systems", Comp. Phys. Comm. **183**, 1760 (2012).


which may also be downloaded from https://arxiv.org/abs/1211.6518 or https://arxiv.org/abs/1110.0573, respectively.

.. _funding-qutip:

Funding
=======
QuTiP is developed under the auspice of the non-profit organizations:

.. _image-numfocus:

.. figure:: figures/NumFocus_logo.png
   :width: 3in
   :figclass: align-center

.. _image-unitaryfund:

.. figure:: figures/unitaryfund_logo.png
   :width: 3in
   :figclass: align-center

QuTiP was partially supported by

.. _image-jsps:

.. figure:: figures/jsps.jpg
   :width: 2in
   :figclass: align-center

.. _image-riken:

.. figure:: figures/riken-logo.png
	:width: 1.5in
	:figclass: align-center

.. _image-korea:

.. figure:: figures/korea-logo.png
	:width: 2in
	:figclass: align-center

.. figure:: figures/inst_quant_sher.png
	:width: 2in
	:figclass: align-center

.. _about:

About QuTiP
===========

Every quantum system encountered in the real world is an open quantum system. For although much care is taken experimentally to eliminate the unwanted influence of external interactions, there remains, if ever so slight, a coupling between the system of interest and the external world. In addition, any measurement performed on the system necessarily involves coupling to the measuring device, therefore introducing an additional source of external influence. Consequently, developing the necessary tools, both theoretical and numerical, to account for the interactions between a system and its environment is an essential step in understanding the dynamics of practical quantum systems.

In general, for all but the most basic of Hamiltonians, an analytical description of the system dynamics is not possible, and one must resort to numerical simulations of the equations of motion. In absence of a quantum computer, these simulations must be carried out using classical computing techniques, where the exponentially increasing dimensionality of the underlying Hilbert space severely limits the size of system that can be efficiently simulated. However, in many fields such as quantum optics, trapped ions, superconducting circuit devices, and most recently nanomechanical systems, it is possible to design systems using a small number of effective oscillator and spin components, excited by a limited number of quanta, that are amenable to classical simulation in a truncated Hilbert space.

The Quantum Toolbox in Python, or QuTiP, is an open-source framework written in the Python programming language, designed for simulating the open quantum dynamics of systems such as those listed above. This framework distinguishes itself from other available software solutions in providing the following advantages:

* QuTiP relies entirely on open-source software.  You are free to modify and use it as you wish with no licensing fees or limitations.

* QuTiP is based on the Python scripting language, providing easy to read, fast code generation without the need to compile after modification.

* The numerics underlying QuTiP are time-tested algorithms that run at C-code speeds, thanks to the `Numpy <https://numpy.org>`_, `Scipy <https://scipy.org>`_, and `Cython <https://cython.org>`_ libraries, and are based on many of the same algorithms used in propriety software.

* QuTiP allows for solving the dynamics of Hamiltonians with (almost) arbitrary time-dependence, including collapse operators.

* Time-dependent problems can be automatically compiled into C++-code at run-time for increased performance.

* Takes advantage of the multiple processing cores found in essentially all modern computers.

* QuTiP was designed from the start to require a minimal learning curve for those users who have experience using the popular quantum optics toolbox by Sze M. Tan.

* Includes the ability to create high-quality plots, and animations, using the excellent `Matplotlib <https://matplotlib.org>`_ package.


For detailed information about new features of each release of QuTiP, see the :ref:`changelog`.

.. _plugin-qutip:

QuTiP Plugins
=============

Several libraries depend on QuTiP heavily making QuTiP a super-library

:Matsubara: `Matsubara <https://matsubara.readthedocs.io/en/latest/>`_ is a plugin to study the ultrastrong coupling regime with structured baths

:QNET: `QNET <https://qnet.readthedocs.io/en/latest/readme.html>`_ is a computer algebra package for quantum mechanics and photonic quantum networks

.. _libraries:

Libraries Using QuTiP
=====================

Several libraries rely on QuTiP for quantum physics or quantum information processing. Some of them are:

:Krotov: `Krotov <https://qucontrol.github.io/krotov/v1.2.0/01_overview.html>`_ focuses on the python implementation of Krotov's method for quantum optimal control

:pyEPR: `pyEPR <https://pyepr-docs.readthedocs.io/en/latest/index.html>`_ interfaces classical distributed microwave analysis with that of quantum structures and hamiltonians by providing easy to use analysis function and automation for the design of quantum chips

:scQubits: `scQubits <https://scqubits.readthedocs.io/en/latest/>`_ is a Python library which provides a convenient way to simulate superconducting qubits by providing an interface to QuTiP

:SimulaQron: `SimulaQron <https://softwarequtech.github.io/SimulaQron/html/index.html>`_ is a distributed simulation of the end nodes in a quantum internet with the specific goal to explore application development

:QInfer: `QInfer <http://qinfer.org/>`_ is a library for working with sequential Monte Carlo methods for parameter estimation in quantum information

:QPtomographer: `QPtomographer <https://qptomographer.readthedocs.io/en/latest/>`_ derive quantum error bars for quantum processes in terms of the diamond norm to a reference quantum channel

:QuNetSim: `QuNetSim <https://tqsd.github.io/QuNetSim/intro.html>`_ is a quantum networking simulation framework to develop and test protocols for quantum networks

:qupulse: `qupulse <https://qupulse.readthedocs.io/en/latest/>`_ is a toolkit to facilitate experiments involving pulse driven state manipulation of physical qubits

:Pulser: `Pulser <https://pulser.readthedocs.io/en/latest/>`_ is a framework for composing, simulating and executing pulse sequences for neutral-atom quantum devices.



Contributing to QuTiP
=====================

We welcome anyone who is interested in helping us make QuTiP the best package for simulating quantum systems.
There are :ref:`detailed instructions on how to contribute code and documentation <development-contributing>` in the developers' section of this guide.
You can also help out our users by answering questions in the `QuTiP discussion mailing list <https://groups.google.com/g/qutip>`_, or by raising issues in `the main GitHub repository <https://github.com/qutip/qutip>`_ if you find any bugs.
Anyone who contributes code will be duly recognized.
Even small contributions are noted.
See :ref:`developers-contributors` for a list of people who have helped in one way or another.

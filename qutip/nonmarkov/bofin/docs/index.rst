.. BoFiN-HEOM documentation master file, created by
   sphinx-quickstart on Mon Oct 19 17:25:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BoFiN-HEOM: HEOM solver for Bosonic and Fermionic systems
=========================================================
BoFiN-HEOM is a "Hierarchical Equations of Motion" solver for quantum systems coupled to Bosonic or Fermionic environments.
The HEOM method was originally developed by Tanimura and Kubo :cite:`Tanimura_1989` .


BoFiN-HEOM is designed to be as generic as possible. It relies on the QuTiP package, and is provided in two versions :

- **BoFiN** : Pure Python version of the HEOM solver. Has a ``BosonicHEOMSolver`` and ``FermionicHEOMSolver`` class. It can be found `here <https://github.com/tehruhn/bofin>`_ .

- **BoFiN-fast** : Hybrid C++ - Python version, of the HEOM solver. Here the backend for RHS construction of the HEOM solver written in C++. It is otherwise completely identical (both in user interface and functionality) to the pure Python version. It can be found `here <https://github.com/tehruhn/bofin_fast>`_ .

It should be noted that the C++ version dramatically speeds up RHS construction, with respect to the Python version. We have noted more than 10x speedup using the C++ version for some hard Fermionic HEOM problems.

The `provided example notebooks <https://github.com/tehruhn/bofin/tree/main/examples>`_ explain some common choices of environmental parameterization. 
See Installation Instructions for details on how to set up both versions.

.. toctree::
   :maxdepth: 2
 
   intro
   install
   bosonic
   fermionic
   dynamics
   steadystate
   zzreferences

   modules
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

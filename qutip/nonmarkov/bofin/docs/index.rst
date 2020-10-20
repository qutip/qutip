.. BoFiN-HEOM documentation master file, created by
   sphinx-quickstart on Mon Oct 19 17:25:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BoFiN-HEOM: A HEOM solver for Bosonic and Fermionic systems
===========================================================
BoFiN-HEOM is a "Hierarchical Equations of Motion" solver for quantum systems coupled to Bosonic or Fermionic environments.
The HEOM method was originally developed by Tanimura and Kubo :cite:`Tanimura_1989` .


BoFiN-HEOM is designed to be as generic as possible, relying on example notebooks to explain how common choices of environmental parameterization is done.   It relies on the QuTiP package, and is provided in two format, this pure Python implementation, and a faster C++ version in a seperate repositary (see installation instructions for details).



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

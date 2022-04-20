**********************************************************
GPU implementation of the Hierarchical Equations of Motion
**********************************************************

.. contents:: Contents
    :local:
    :depth: 3

The Hierarchical Equations of Motion (HEOM) method is a non-perturbative
approach to simulate the evolution of the density matrix of dissipative quantum
systems. The underlying equations are a system of coupled ODEs which can be run
on a GPU. This will allow the study of larger systems as discussed in [1]_. The
goal of this project would be to extend QuTiP's HEOM method [2]_ and implement
it on a GPU.

Since the method is related to simulating large, coupled ODEs, it can also be
quite general and extended to other solvers.

Expected outcomes
=================

* A version of HEOM which runs on a GPU.
* Performance comparison with the CPU version.
* Implement dynamic scaling.

Skills
======

* Git, python and familiarity with the Python scientific computing stack
* CUDA and OpenCL knowledge

Difficulty
==========

* Hard

Mentors
=======

* Neill Lambert (nwlambert@gmail.com)
* Alex Pitchford (alex.pitchford@gmail.com)
* Shahnawaz Ahmed (shahnawaz.ahmed95@gmail.com)
* Simon Cross (hodgestar@gmail.com)

References
==========

.. [1] https://pubs.acs.org/doi/abs/10.1021/ct200126d?src=recsys&journalCode=jctcce
.. [2] https://arxiv.org/abs/2010.10806

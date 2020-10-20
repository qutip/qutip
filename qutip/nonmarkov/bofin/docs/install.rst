############
Installation
############

We have developed two packaged versions of the HEOM solver : 

- **BoFiN** : Pure Python version of the HEOM solver. Has a ``BosonicHEOMSolver`` and ``FermionicHEOMSolver`` class. It can be found `here <https://github.com/tehruhn/bofin>`_ .

- **BoFiN-fast** : Hybrid C++ - Python version, of the HEOM solver. Here the backend for RHS construction of the HEOM solver written in C++. It is otherwise completely identical (both in user interface and functionality) to the pure Python version. It can be found `here <https://github.com/tehruhn/bofin_fast>`_ .

The following sections explain how to set up both versions, and common dependencies.

Installing dependencies & setting up
====================================

The core requirements are ``numpy``, ``scipy``, ``cython`` and ``qutip``.
For uniformity across platforms, we recommend using Conda environments to keep the setup clean, and to install dependencies painlessly (since ``pip install`` is known to have issues on Mac OS). 
Once you have Conda installed, make a fresh Python 3 environment called ``bofin_env``, and then switch to it::

    conda create -n bofin_env python=3.8
    conda activate bofin_env

In your ``bofin_env`` environment, install requirements using::

    conda install numpy scipy cython
    conda install -c conda-forge qutip

Also, ``matplotlib`` is required for visualizations.
This will ensure painless setup across Windows, Linux and Mac OS.

Installing the Python version
=============================

Clone the BoFiN repository given `here <https://github.com/tehruhn/bofin>`_ using ``git clone``.

Once you have the dependencies installed, from the parent repository folder, run the following command::

    pip3 install -e .

This will install the pure Python version of the Bosonic HEOM Solver.

Installing the C++ version
==========================

Clone the BoFiN-fast repository given `here <https://github.com/tehruhn/bofin_fast>`_ using ``git clone``.

Once you have the dependencies installed, from the parent repository folder, run the following commands::

    python3 setup.py build_ext --inplace
    pip3 install -e .


This installs the hybrid Python - C++ version of the HEOM solvers. These are identical in usage and functionality to the Python solvers.
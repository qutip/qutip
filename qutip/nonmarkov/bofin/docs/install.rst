############
Installation
############

We have developed two packaged versions of the HEOM solver : 

- BoFiN : Pure Python version of the HEOM solver. Has a `BosonicHEOMSolver` and `FermionicHEOMSolver`.
- BoFiN-fast : Hybrid C++ - Python version, with backend for RHS construction of the HEOM solver written in C++. Otherwise completely identical in user interface and functionality to the pure Python version.

It should be noted that the C++ version dramatically speeds up RHS construction, with respect to the Python version. 
We have noted more than 10x speedup using the C++ version for some hard Fermionic HEOM problems. 
If you run the code on hard-to-compute systems, we recommend you set up the C++ version as given here 'Bofin Fast <https://github.com/tehruhn/bofin_fast>'


Installing dependencies & setting up
====================================

For uniformity across platforms, we recommend using Conda environments to keep the setup clean, and to install dependencies painlessly (since `pip install` is known to have issues on Mac OS). 
Once you have Conda installed, make a fresh Python 3 environment called `bofin_env`, and then switch to it::

    conda create -n bofin_env python=3.8
    conda activate bofin_env

In your `bofin_env` environment, install requirements using::

    conda install numpy scipy cython
    conda install -c conda-forge qutip


This will ensure painless setup across Windows, Linux and Mac OS.

Installing the BoFiN-HEOM package
=================================

Clone the repository using `git clone`.

Once you have the dependencies installed, from the parent repository folder, run the following command::

    pip3 install -e .

This will install the pure Python version of the Bosonic HEOM Solver.

See 'Bofin Fast <https://github.com/tehruhn/bofin_fast>' for information on installing the C++ version.
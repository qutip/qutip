# BoFiN-HEOM (Python version) : A Bosonic and Fermionic hierarchical-equations-of-motion library with applications in light-harvesting, quantum control, and single-molecule electronics

Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alexander Pitchford, Franco Nori 

## Abstract

The “hierarchical equations of motion” (HEOM) method is a powerful numerical approach to solve the dynamics and steady-state of a quantum system coupled to a non-Markovian and non-perturbative environment. Originally developed in the context of physical chemistry, it has also been extended and applied to problems in solid-state physics, optics, single-molecule electronics,and biological physics. Here we present a numerical library in Python, integrated with the powerful QuTiP platform, which implements the HEOM for both Bosonic and Fermionic environments. Wedemonstrate it’s utility with a series of examples.  For the Bosonic case, we present examples for fitting arbitrary spectral densities, modelling a Fenna-Matthews-Olsen photosynthetic complex,and simulating dynamical decoupling of a spin from its environment.  For the Fermionic case, we present an integrable single-impurity example, used as a benchmark of the code, and a more complex example of an impurity strongly coupled to a single vibronic mode, with applications in single-molecule electronics.

## Python HEOM Solver

We have developed two packaged versions of the HEOM solver : 

- **BoFiN** : Pure Python version of the HEOM solver. Has a `BosonicHEOMSolver` and `FermionicHEOMSolver`.
- **BoFiN-Fast** : Hybrid C++ - Python version, with backend for RHS construction of the HEOM solver written in C++. Otherwise completely identical in user interface and functionality to the pure Python version.

This repository contains the pure Python version of the solver. It should be noted that the C++ version dramatically speeds up RHS construction, with respect to the Python version. We have noted more than 10x speedup using the C++ version for some hard Fermionic HEOM problems. 

### FOR COMPUTATIONALLY INTENSIVE CASES

 If you run the code on hard-to-compute systems, we recommend you set up the C++ version as given here : https://github.com/tehruhn/bofin_fast


## Installing dependencies & setting up

For uniformity across platforms, we recommend using Conda environments to keep the setup clean, and to install dependencies painlessly (since `pip install` is known to have issues on Mac OS). Once you have Conda installed, make a fresh Python 3 environment called `bofin_env`, and then switch to it :

```
conda create -n bofin_env python=3.8
conda activate bofin_env
```

In your `bofin_env` environment, install requirements using :
```
conda install numpy scipy cython
conda install -c conda-forge qutip
```

This will ensure painless setup across Windows, Linux and Mac OS.

## Installing the BoFiN-HEOM package

Clone the repository using `git clone`.

Once you have the dependencies installed, from the parent repository folder, run the following command :
```
pip3 install -e .
```
This will install the pure Python version of the Bosonic HEOM Solver.

## Usage example

```
# import the pure Python Bosonic HEOM Solver
from bofin.heom import BosonicHEOMSolver as BosonicHEOMSolverPy
```

## Documentation

The documentation can be found at https://bofin-heom.readthedocs.io/en/main/ .

To build the documentation locally, you will need `sphinx` and, `sphinx_rtd_theme`:
```
cd docs/
make html
```

## Example notebooks

There are several example notebooks illustrating usage of the code, in the `examples` folder.

## Running tests

To run tests using `pytest` , in the parent repository, run :
```
pytest -v
```
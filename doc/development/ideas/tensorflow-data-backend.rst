***********************
TensorFlow Data Backend
***********************

.. contents:: Contents
    :local:
    :depth: 3

.. note::
    This project was completed as part of GSoC 2021 [3]_.

QuTiP's data layer provides the mathematical operations needed to work with
quantum states and operators, i.e. ``Qobj``, inside QuTiP. As part of Google
Summer of Code 2020, the data layer was rewritten to allow new backends to
be added more easily and for different backends to interoperate with each
other. Backends using in-memory spares and dense matrices already exist,
and we would like to add a backend that implements the necessary operations
using TensorFlow [1]_.

Why a TensorFlow backend?
-------------------------

TensorFlow supports distributing matrix operations across multiple GPUs and
multiple machines, and abstracts away some of the complexities of doing so
efficiently. We hope that by using TensorFlow we might enable QuTiP to scale
to bigger quantum systems (e.g. more qubits) and decrease the time taken to
simulate them.

There is particular interest in trying the new backend with the
BoFiN HEOM (Hierarchical Equations of Motion) solver [2]_.

Challenges
----------

TensorFlow is a very different kind of computational framework to the existing
dense and sparse matrix backends. It uses flow graphs to describe operations,
and to work efficiently. Ideally large graphs of operations need to be
executed together in order to efficiently compute results.

The QuTiP data layer might need to be adjusted to accommodate these
differences, and it is possible that this will prove challenging or even
that we will not find a reasonable way to achieve the desired performance.

Expected outcomes
=================

* Add a ``qutip.core.data.tensorflow`` data type.
* Implement specialisations for some important operations (e.g. ``add``,
  ``mul``, ``matmul``, ``eigen``, etc).
* Write a small benchmark to show how ``Qobj`` operations scale on the new
  backend in comparison to the existing backends. Run the benchmark both
  with and without using a GPU.
* Implement enough for a solver to run on top of the new TensorFlow data
  backend and benchmark that (stretch goal).

Skills
======

* Git, Python and familiarity with the Python scientific computing stack
* Familiarity with TensorFlow (beneficial, but not required)
* Familiarity with Cython (beneficial, but not required)

Difficulty
==========

* Medium

Mentors
=======

* Simon Cross (hodgestar@gmail.com)
* Jake Lishman (jake@binhbar.com)
* Alex Pitchford (alex.pitchford@gmail.com)

References
==========

.. [1] https://www.tensorflow.org/
.. [2] https://github.com/tehruhn/bofin
.. [3] https://github.com/qutip/qutip-tensorflow/

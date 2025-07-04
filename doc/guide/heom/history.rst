########################
Previous implementations
########################

The current HEOM implementation in QuTiP is the latest in a succession of HEOM
implementations by various contributors:


HSolverDL
---------

The original HEOM solver was implemented by Neill Lambert, Anubhav Vardhan,
and Alexander Pitchford. In QuTiP 4.7 it was still available as
``qutip.solve.nonmarkov.dlheom_solver.HSolverDL`` but the legacy implementation
was removed in QuTiP 5.

It only directly provided support for the Drude-Lorentz bath although there was
the possibility of sub-classing the solver to implement other baths.

A compatible interface using the current implementation is still available
under the same name in :class:`qutip.solver.heom.HSolverDL`.


BoFiN-HEOM
----------

BoFiN-HEOM (the bosonic and fermionic HEOM solver) was a much more
flexible re-write of the original QuTiP ``HSolverDL`` that added support for
both bosonic and fermionic baths and for baths to be specified directly via
their correlation function expansion coefficients. Its authors were
Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, and Alexander Pitchford.

BoFiN was written outside of QuTiP and is can still be found in its original
repository at https://github.com/tehruhn/bofin.

The construction of the right-hand side matrix for BoFiN was slow, so
BoFiN-fast, a hybrid C++ and Python implementation, was written that performed
the right-hand side construction in C++. It was otherwise identical to the
pure Python version. BoFiN-fast can be found at
https://github.com/tehruhn/bofin_fast.

BoFiN also came with an extensive set of example notebooks that are available
at https://github.com/tehruhn/bofin/tree/main/examples.


Current implementation
----------------------

The current implementation is a rewrite of BoFiN in pure Python. It's right-hand
side construction has similar speed to BoFiN-fast, but is written in pure
Python. Built-in implementations of a variety of different baths are provided,
and a single solver is used for both fermionic and bosonic baths. Multiple baths
of either the same kind, or a mixture of fermionic and bosonic baths, may be
specified in a single problem, and there is good support for working with the
auxiliary density operator (ADO) state and extracting information from it.

The code was written by Neill Lambert and Simon Cross.

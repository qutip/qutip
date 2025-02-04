############
Introduction
############

The Hierarchical Equations of Motion (HEOM) method was originally developed by
Tanimura and Kubo :cite:`Tanimura_1989` in the context of physical chemistry to
''exactly'' solve a quantum system in contact with a bosonic environment,
encapsulated in the Hamiltonian:

.. math::

	H = H_s + \sum_k \omega_k a_k^{\dagger}a_k + \hat{Q} \sum_k g_k \left(a_k + a_k^{\dagger}\right).

As in other solutions to this problem, the properties of the bath are
encapsulated by its temperature and its spectral density,

.. math::

    J(\omega) = \pi \sum_k g_k^2 \delta(\omega-\omega_k).

In the HEOM, for bosonic baths, one typically chooses a Drude-Lorentz spectral
density:

.. math::

    J_D = \frac{2\lambda \gamma \omega}{\gamma^2 + \omega^2},

or an under-damped Brownian motion spectral density:

.. math::

    J_U = \frac{\lambda^2 \Gamma \omega}{(\omega_c^2 - \omega^2)^2 + \Gamma^2 \omega^2}.

Given the spectral density, the HEOM requires a decomposition of the bath
correlation functions in terms of exponentials.
Generally, such decompositions can be generated using the methods available in the :ref:`environment module <environments guide>`.
We will go into some more detail in :doc:`bosonic`, describe
how this is done with code examples, and how these expansions are passed to the
solver.

In addition to support for bosonic environments, QuTiP also provides support for
fermionic environments which is described in :doc:`fermionic`.

Both bosonic and fermionic environments are supported via a single solver,
:class:`.HEOMSolver`, that supports solving for both dynamics and steady-states.

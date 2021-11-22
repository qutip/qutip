############
Introduction
############


The HEOM method was originally developed in the context of Physical Chemistry to ''exactly'' solve a quantum system in contact with a bosonic environment, encapsulated in the Hamiltonian

.. math::

	H = H_s + \sum_k \omega_k a_k^{\dagger}a_k + \hat{Q} \sum_k g_k \left(a_k + a_k^{\dagger}\right).

As in other solutions to this problem, the properties of the bath are encapsulated by its temperature and its spectral density,

.. math::

    J(\omega) = \pi \sum_k g_k^2 \delta(\omega-\omega_k).

In the HEOM, for bosonic baths, we typically choose a Drude-Lorentz spectral density

.. math:: 

    J_D = \frac{2\lambda \gamma \omega}{(\gamma^2 + \omega^2)}

or an under-damped Brownian motion spectral density

.. math::

    J_U = \frac{\alpha^2 \Gamma \omega}{[(\omega_c^2 - \omega^2)^2 + \Gamma^2 \omega^2]}.
    
    
Other cases are usually approached with fitting.

Given the spectral density, the HEOM needs a decomposition of the bath correlation functions in terms of exponentials. In the Matsubara and Pade sections we describe how this is done with code examples, and how these are passed to the solver.

In addition to the Bosonic case we also provide a solver for Feriomic environments.

The two options are dealt with by two different classes, ``BosonicHEOMSolver`` and ``FermionicHEOMSolver``.

Each have associated ODE solvers for dynamics and steady-state solutions, as required.


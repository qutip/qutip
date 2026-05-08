.. _dysolve:

*******
Dysolve
*******

The ``dysolve`` solver [1]_ provides an efficient way to compute the time evolution of closed quantum systems under high-frequency drives.
It uses a Dyson series expansion in the interaction picture, offering a significant performance advantage over standard methods like :func:`sesolve`
when the drive frequency is large compared to the system's internal energy scales.

The Time Evolution Operator
===========================
The time evolution of a closed quantum system is governed by the Schrödinger equation (:math:`\hbar = 1`):

.. math::

  \displaystyle i \frac{d}{dt}\left|\psi(t)\right> = H(t)\left|\psi(t)\right>


For a time-dependent Hamiltonian, the solution is expressed via the time-evolution operator (propagator) :math:`U(t_f, t_i)`,
which maps an initial state to a final state: :math:`\left|\psi(t_f)\right> = U(t_f, t_i)\left|\psi(t_i)\right>`.
Generally, this operator takes the form of a time-ordered exponential:

.. math::

  \displaystyle U(t_f, t_i) = \mathcal{T} \exp\left(-i \int_{t_i}^{t_f} H(t) dt \right)


where :math:`\mathcal{T}` is the time-ordering operator.
While standard solvers compute this by numerically integrating the ODE,
``dysolve`` uses a perturbative expansion that is particularly powerful for oscillatory drives.


.. _DysolveMethod:

The Dyson Series Method
=======================

``dysolve`` is designed for Hamiltonians that can be decomposed into a static part
:math:`H_0` and a sum of periodic perturbations:

.. math::

    H(t) = H_0 + \sum_j V_j(t) = H_0 + \sum_j \mathcal{E}_j(t) X_j e^{i \omega_j t}

where :math:`X_j` are drive operators, :math:`\omega_j` are frequencies,
and :math:`\mathcal{E}_j(t)` are (potentially) slow-moving envelopes.

By moving to the interaction picture defined by :math:`H_0`,
the propagator over a small time step :math:`\delta t` can be expanded as a Dyson series:

.. math::

    U(t+\delta t, t) \approx \sum_{n=0}^{r} U^{(n)}(t + \delta t, t)

The expansion terms :math:`U^{(n)}` are computed analytically.
A key advantage of this method is that for a constant envelope :math:`\mathcal{E}_j`,
the heavy lifting of the expansion—specifically the multidimensional integrals—depends only on
:math:`\delta t` and the frequencies, not on the absolute time :math:`t`.

Why use Dysolve?
----------------

1. **High-Frequency Efficiency**: Standard ODE solvers require a time step much smaller than the shortest period in the system (:math:`\delta t \ll 1/\omega_{max}`).
   In contrast, the Dyson expansion converges **faster** as the frequency increases.
2. **Memoization**: Because the core tensors (:math:`S^{(n)}`) are independent of the current time :math:`t`, they are computed once and cached.
   This makes long-time simulations or calculations with many time-steps extremely fast after the first step.

.. _dysolve_usage:

Using Dysolve
=============

The solver is accessed via the :func:`dysolve` function or the :class:`Dysolve` class for more granular control.

**Key Options:**

* ``order``: The truncation order of the Dyson series (default is 4).
  Higher orders increase accuracy but require more precomputation.
* ``step_size``: The time increment :math:`\delta t`.
  The envelope :math:`\mathcal{E}(t)` is assumed to be constant over this interval.
* ``eigen``: How to get the Hamiltonian in the interaction picture.
  If true, the Hamiltonian will be diagonalized.
  If false, the non-diagonal parts will be computed as a drive with frequency of zeros.
  Precomputation is much slower with full diagonalization, but the numerical error is much smaller.

.. _dysolve_code_example:

Code Example
============

In this example, we compare ``dysolve`` to the standard ``sesolve`` for a
Rabi oscillation under a high-frequency drive.

.. code-block:: python

    import numpy as np
    from qutip import sigmaz, sigmax, basis, dysolve, sesolve, CoreOptions

    # Parameters
    omega_q = 1.0 * 2 * np.pi
    omega_d = 1000.0 * 2 * np.pi
    A = 0.5
    tlist = np.linspace(0, 10, 201)

    # Hamiltonian components
    H0 = 0.5 * omega_q * sigmaz()
    # Drive format: (Operator, Frequency, Form)
    drives = [(A * sigmax(), omega_d, "cos")]

    # Initial state
    psi0 = basis(2, 0)

    # Solve using Dyson series
    result_dy = dysolve(
        H0, drives, psi0, tlist, 
        options={"order": 4, "step_size": 0.1}
    )

    # Solve using sesolve (for comparison)
    H_td = [H0, [sigmax(), lambda t, args: A * np.cos(omega_d * t)]]
    result_se = sesolve(H_td, psi0, tlist)

    with CoreOptions(atol=1e-7, rtol=1e-7):
      print(result_se.states == result_dy.states)

References
==========
.. [1] Ross Shillito, Jonathan A. Gross, Agustin Di Paolo, Élie Genois, and Alexandre Blais. Fast and differentiable simulation of driven quantum systems. *Physical Review Research*, 3(3), September 2021. https://arxiv.org/abs/2012.09282

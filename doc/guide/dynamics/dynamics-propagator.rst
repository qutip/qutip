.. _propagator:

*********************
Computing propagators
*********************

Sometime the evolution of a single state is not sufficient and the full propagator
is desired. QuTiP has the :func:`.propagator` function to compute them:

.. code-block::

    >>> H = sigmaz() + np.pi *sigmax()
    >>> psi_t = sesolve(H, basis(2, 1), [0, 0.5, 1]).states
    >>> prop = propagator(H, [0, 0.5, 1])

    >>> print((psi_t[1] - prop[1] @ basis(2, 1)).norm())
    2.455965272327082e-06

    >>> print((psi_t[2] - prop[2] @ basis(2, 1)).norm())
    2.0071900004562142e-06


The first argument is the Hamiltonian, any time dependent system format is
accepted. The function also accepts an optional `c_ops` argument for collapse operators.
When used, a propagator for density matrices is computed:
:math:`\rho(t) = U(t)(\rho(0))`:

.. code-block::

    >>> rho_t = mesolve(H, fock_dm(2, 1), [0, 0.5, 1], c_ops=[sigmam()]).states
    >>> prop = propagator(H, [0, 0.5, 1], c_ops=[sigmam()])

    >>> print((rho_t[1] - prop[1](fock_dm(2, 1))).norm())
    7.23009476734681e-07

    >>> print((rho_t[2] - prop[2](fock_dm(2, 1))).norm())
    1.2666967766644768e-06


The propagator function is also available as a class:

.. code-block::

    >>> U = Propagator(H, c_ops=[sigmam()])

    >>> state_0_5 = U(0.5)(fock_dm(2, 1))
    >>> state_1 = U(1., t_start=0.5)(state_0_5)

    >>> print((rho_t[1] - state_0_5).norm())
    7.23009476734681e-07

    >>> print((rho_t[2] - state_1).norm())
    8.355518501351504e-07

The :obj:`.Propagator` can take ``options`` and ``args`` as a solver instance.

.. _propagator_solver:


Using a solver to compute a propagator
======================================

Many solvers accept an operator as the initial state. When an identity matrix is
passed as the initial state, the propagator is computed. This can be used to compute
a propagator for Bloch-Redfield or Floquet equations:

.. code-block::

  >>> delta = 0.2 * 2*np.pi
  >>> eps0 = 1.0 * 2*np.pi
  >>> gamma1 = 0.5

  >>> H = - delta/2.0 * sigmax() - eps0/2.0 * sigmaz()

  >>> def ohmic_spectrum(w):
  >>>   if w == 0.0: # dephasing inducing noise
  >>>     return gamma1
  >>>   else: # relaxation inducing noise
  >>>     return gamma1 / 2 * (w / (2 * np.pi)) * (w > 0.0)

  >>> prop = brmesolve(H, qeye(2), [0, 1], a_ops=[[sigmax(), ohmic_spectrum]]).final_state

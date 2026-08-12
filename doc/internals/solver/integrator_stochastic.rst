.. _integrator_stochastic_internal:

Stochastic Integrator Architecture
##################################

Stochastic Differential Equations (SDEs) are used to simulating open quantum dynamics under continuous measurement.
In QuTiP, SDE integrators inherit from :class:`SIntegrator`,
which extends the base :class:`Integrator` interface to handle stochastic noise processes.

Key Differences from Standard Integrators
=========================================

Attributes:

.. attribute:: SIntegrator.rhs_format

   Supported formats for specifying the right-hand side (RHS) of the stochastic equation:

   * ``"SDESystem"``:
     A class deriving from :class:`StochasticSystem`.
     Exposes ``drift`` and ``diffusion`` methods representing the deterministic and stochastic parts of the equation, respectively.
     Used for explicit solvers (e.g., :class:`EulerSODE`, :class:`PlatenSODE`).
   * ``"SDETaylorSystem"``:
     A class deriving from :class:`TaylorStochasticSystem`.
     In addition to ``drift`` and ``diffusion``, it provides Ito-Taylor expansion derivatives up to order 1.5.
   * ``"system"``:
     An object containing system operators (e.g., ``H``, ``c_ops``, ``sc_ops``)
     required to construct the equations dynamically.

.. attribute:: SIntegrator._support_measurement_noise

   A boolean flag indicating whether measurement noise is supported
   (e.g., when Wiener processes correspond directly to experimental measurements).

.. attribute:: SIntegrator.N_dw

   An integer specifying the number of noise variables required per step for each diffusion operator.
   Standard explicit integrators require $N_{dw} = 1$, whereas higher-order Taylor integrators (such as order 1.5) may require more.


Methods
-------

.. method:: SIntegrator.set_state(t, state, wiener)

   Sets the initial state and noise process.

   :param float t: Initial time.
   :param Data state: Initial state in QuTiP data structure format (e.g., :class:`~qutip.core.data.Dense`).
   :param Wiener wiener: A :class:`Wiener` object managing noise generation. Saved to ``self.wiener``.

.. method:: SIntegrator.get_state()

   Returns the current integrator state as a tuple ``(t, state, wiener)``,
   where ``wiener`` is the instance attached via :meth:`set_state`.


Helper Classes
##############

:class:`Wiener`
===============
Represent the noise.
The only method used by SIntegrator is :method:`Wiener.dW(t, N)`:

Return an array of noise for N steps starting at ``t``.
The noise is of shape ``(N, N_dw, num_diffusion)``.
The noise Gaussian noise scaled with ``dt**0.5``.
The ``t`` is expected to be a multiple of the ``dt`` options and is the time at the start of the step.

The :class:`Wiener` class generates Gaussian stochastic noise increments.
The main method called by :class:`SIntegrator` is:

.. method:: Wiener.dW(t, N)

   Returns a array of shape ``(N, N_dw, num_diffusion)`` of Wiener increments
   for $N$ integration steps starting from time $t$.
   The noise is normal random variables scaled by $\sqrt{dt}$.
   $t$ must be an integer multiple of the configured step size ``dt``.

Also available:

.. method:: Wiener.__call__(t)

   Return the continuous Wiener process at time ``t``.


:
Stochastic Systems
##################

Stochastic system classes act as structured containers for the drift vector and diffusion matrix field.

1. **:class:`BaseStochasticSystem`**
   The abstract base class establishing the expected interface:

   * ``drift(t: float, state: Data) -> Data``: Computes the deterministic drift vector $a(t, x)$.
   * ``diffusion(t: float, state: Data) -> list[Data]``: Computes the stochastic diffusion vectors $b^i(t, x)$ for each noise source $i$.
   * ``_shift(t: float, state: Data) -> list[float]``: Internal helper used by ``run_from_experiment`` to calculate the offset between pure Wiener noise and continuous experimental measurement outputs.

2. **:class:`StochasticSystem`**
   A concrete wrapper class for custom system functions.

3. **:class:`TaylorStochasticSystem`**
   Extends :class:`BaseStochasticSystem` by adding Ito-Taylor differential operators. To minimize redundant calculations, state evaluation is cached internally via state setters:

   * ``set_state(t: float, state: Data)``: Evaluates and caches state dependencies.
   * ``a()``: Returns the drift term evaluated at the cached state.
   * ``bi(i: int)``: Returns the $i$-th diffusion term evaluated at the cached state.
   * Differential operators:

     * ``Libj(i, j)``: $\sum_n b_n^i \frac{\partial b^j}{\partial x_n}$
     * ``Lia(i)``: $\sum_n b_n^i \frac{\partial a}{\partial x_n}$
     * ``L0bi(i)``: Drift-action operator on the $i$-th diffusion term.
     * ``LiLjbk(i, j, k)``: Second-order iterated diffusion operator.
     * ``L0a()``: Drift-action operator on the drift term.

4. **:class:`StochasticClosedSystem`**
   Optimized C-extension system implementation for the Stochastic Schrödinger Equation solver (:class:`~qutip.solver.stochastic.SSESolver`).

5. **:class:`StochasticOpenSystem`**
   Optimized C-extension system implementation for the Stochastic Master Equation solver (:class:`~qutip.solver.stochastic.SMESolver`).


Building Custom Stochastic Evolutions
######################################

For custom SDE forms not directly covered by standard solvers, QuTiP's internal SDE integrators can be used directly.

Explicit Methods (Derivative-Free)
==================================

Methods such as :class:`EulerSODE`, :class:`PlatenSODE`, and :class:`Explicit1_5_SODE` do not require analytical derivatives:

.. code-block:: python

    import numpy as np
    import qutip
    from qutip.core.data import Data
    from qutip.solver.sode import EulerSODE
    from qutip.solver.sode._noise import Wiener
    from qutip.solver.sode.ssystem import StochasticSystem

    # Prepare system operators
    H = qutip.rand_herm(4)
    a_op = qutip.destroy(4)
    num_op = qutip.num(4)

    # Extract raw data objects
    L = (-1j * H - 0.5 * a_op.dag() @ a_op).data
    ad = a_op.dag().data
    a = a_op.data

    psi0 = qutip.basis(4, 3).data
    options = {"dt": 0.001}

    # Define RHS drift and diffusion functions
    def drift(t: float, state: Data) -> Data:
        return L @ state

    def diffusion(t: float, state: Data) -> list[Data]:
        return [a @ state, ad @ state]

    # Initialize system, noise, and integrator
    system = StochasticSystem(drift, diffusion, num_diffusion=2)
    wiener = Wiener(t0=0, dt=options["dt"], generator=np.random.default_rng(0), num_diffusion=2)

    SDE = EulerSODE(system, options)
    SDE.set_state(0, psi0, wiener)

    # Run trajectory integration
    expect = []
    for t in np.linspace(0, 1, 11):
        # Note: target times t must align with integer multiples of dt
        t_out, state_t, _ = SDE.integrate(t)
        assert np.allclose(t, t_out)
        expect.append(qutip.expect(num_op, qutip.Qobj(state_t)))

    print(expect)


Taylor Methods (Derivative-Based)
=================================

Solvers requiring Ito-Taylor expansions (e.g., :class:`Milstein_SODE`, :class:`Taylor1_5_SODE`) require subclasses of :class:`TaylorStochasticSystem`:

.. code-block:: python

    import numpy as np
    import qutip
    from qutip.solver.sode import Milstein_SODE
    from qutip.solver.sode._noise import Wiener
    from qutip.solver.sode.ssystem import TaylorStochasticSystem

    # Prepare system operators
    H = qutip.rand_herm(4)
    a_op = qutip.destroy(4)
    num_op = qutip.num(4)

    L = (-1j * H - 0.5 * a_op.dag() @ a_op).data
    a2 = (a_op @ a_op).data
    a = a_op.data

    psi0 = qutip.basis(4, 3).data
    options = {"dt": 0.001}


    class CustomTaylorSystem(TaylorStochasticSystem):
        def a(self):
            return L @ self.state

        def bi(self, i: int):
            return [a, a2][i] @ self.state

        def Libj(self, i: int, j: int):
            r"""
            First-order diffusion derivative operator acting on a diffusion term:

            .. math::

               (L_i b^j)_\mu = \sum_n b^i_n \frac{\partial b^j_\mu}{\partial x_n}
            """
            # Note: Solvers assume commutative diffusion terms (Libj == Ljbi)
            opers = [a, a2]
            return opers[i] @ (opers[j] @ self.state)

        # Note: Additional higher-order derivatives (Lia, L0bi, etc.)
        # must be defined if using Taylor1_5_SODE.


    # Initialize system, noise, and integrator
    system = CustomTaylorSystem(num_diffusion=2)
    wiener = Wiener(t0=0, dt=options["dt"], generator=np.random.default_rng(0), num_diffusion=2)

    SDE = Milstein_SODE(system, options)
    SDE.set_state(0, psi0, wiener)

    # Run trajectory integration
    expect = []
    for t in np.linspace(0, 1, 11):
        t_out, state_t, _ = SDE.integrate(t)
        assert np.allclose(t, t_out)
        expect.append(qutip.expect(num_op, qutip.Qobj(state_t)))

    print(expect)



Stochastic Systems
##################
The systems set of classes are containers for the drift,
diffusion pair for the stochastic evolution.

There are 5 classes included:
- BaseStochasticSystem
  Empty classes, provide the generic signature used by SDE integrators:

  ``drift(t: float, state: Data) -> Data``:
  the deterministic part of the evolutions

  ``diffusion(t: float, state: Data) -> list[Data]``
  Random part of the evolution.
  Each terms following it's own wiener noise.

  ``_shift(self, t, Data state) -> list[int]``
  Helper function for "run_from_experiment" compute the difference between the
  Wiener noise and the experimental measurement.

- StochasticSystem(BaseStochasticSystem)
  Helper class for custom system.

- TaylorStochasticSystem(BaseStochasticSystem)
  System with the derivative also available.
  It has now method for each derivatives:
  - set_state(t: float, state: Data)
    Store the state internally
  - a(): drift
  - bi(i): diffusion
  - derivatives:
    - Libj(i, j)
    - Lia(i)
    - L0bi(i)
    - LiLjbk(i, j, k)
    - L0a()
  Storing the state internally allow to reuse intermediate computation between
  derivatives calls.

- StochasticClosedSystem(BaseStochasticSystem)
  Hard coded system for SSESovler

- StochasticOpenSystem(TaylorStochasticSystem)
  Hard coded system for SMESovler

Building your own Stochastic evolution
######################################

The SDE solver in qutip are focused on solving the standard stochastic evolution.
However python has few SDE solvers available and something some modification to the equation might be needed.
In that situation, it is possible to use QuTiP's SDE directly.
As this is not the expected use cases, the interface is not ideal, but it's usable.


For integration methods that do not uses derivative (EulerSODE, PlatenSODE, Explicit1_5_SODE):

.. code-block: python

    import qutip
    import numpy as np
    from qutip.solver.sode import EulerSODE, Milstein_SODE
    from qutip.solver.sode._noise import Wiener
    from qutip.solver.sode.ssystem import StochasticSystem, TaylorStochasticSystem
    from qutip.core.data import Data

    # Prepare operators
    H = qutip.rand_herm(4)
    a = qutip.destroy(4)
    num = qutip.num(4)

    # Extract raw matrix data
    L = (-1j * H - a @ a.dag()).data
    ad = a.dag().data
    a = a.data

    psi0 = qutip.basis(4, 3).data
    options = {"dt": 0.001}

    # Define RHS functions
    def drift(t: float, state: Data) -> Data:
        return L @ state

    def diffusion(t: float, state: Data) -> list[Data]:
        return [a @ state, ad @ state]

    # Create the system
    system = StochasticSystem(drift, diffusion, num_diffusion=2)
    wiener = Wiener(t0=0, dt=options["dt"], generator=np.random.default_rng(0), num_diffusion=2)

    # Create and initiate the integrator
    SDE = EulerSODE(system, options)
    SDE.set_state(0, psi0, wiener)

    # Run the evolution
    expect = []
    for t in np.linspace(0, 1, 11):
        # t's must be multiple of dt
        t_out, state_t, noise = SDE.integrate(t)
        assert np.allclose(t, t_out)
        expect.append(qutip.expect(num, qutip.Qobj(state_t)))

    # Check results
    print(expect)


And for Taylor method using derivatives:

.. code-block: python

    import qutip
    import numpy as np
    from qutip.solver.sode import EulerSODE, Milstein_SODE
    from qutip.solver.sode._noise import Wiener
    from qutip.solver.sode.ssystem import StochasticSystem, TaylorStochasticSystem
    from qutip.core.data import Data

    # Prepare operators
    H = qutip.rand_herm(4)
    a = qutip.destroy(4)
    num = qutip.num(4)

    # Extract raw matrix data
    L = (-1j * H - a @ a.dag()).data
    a2 = (a @ a).data
    a = a.data

    psi0 = qutip.basis(4, 3).data
    options = {"dt": 0.001}


    class MySystem(TaylorStochasticSystem):
        def a(self):
            return L @ self.state

        def bi(self, i: int):
            return [a, a2][i] @ self.state

        def Libj(self, i: int, j: int):
            r"""
            First-order diffusion derivative operator acting on a diffusion term.

            $$(L_i b^j)_\mu = \sum_n b^i_n \frac{\partial b^j_\mu}{\partial x_n}$$
            """
            # All taylor integrator expect commutative diffusion terms
            # Libj == Ljbi
            opers = [a, a2]
            return opers[i] @ (opers[j] @ self.state)

        # Higher order derivative are needed for Taylor1_5_SODE

    # Create the system
    system = MySystem(num_diffusion=2)
    wiener = Wiener(t0=0, dt=options["dt"], generator=np.random.default_rng(0), num_diffusion=2)

    # Create and initiate the integrator
    SDE = Milstein_SODE(system, options)
    SDE.set_state(0, psi0, wiener)

    # Run the evolution
    expect = []
    for t in np.linspace(0, 1, 11):
        # t's must be multiple of dt
        t_out, state_t, noise = SDE.integrate(t)
        assert np.allclose(t, t_out)
        expect.append(qutip.expect(num, qutip.Qobj(state_t)))

    # Check results
    print(expect)

.. _integrator_internal:

Integrator Architecture and Design
##################################

The numerical solution of differential equations is a cornerstone of QuTiP's
time-evolution solvers. Because different quantum systems present varying behaviours,
no single numerical algorithm is optimal for all scenarios.

The :class:`Integrator` framework provides a unified, backend-agnostic interface
that wraps external Ordinary Differential Equation (ODE) libraries. This design
allows solvers to select routines from multiple distinct libraries through a single,
interchangeable class.


Motivation
==========

While Python packages like SciPy offer a robust collection of integration routines,
they present several architectural limitations when applied directly to quantum dynamics:

* **Inconsistent API Interfaces**: Across SciPy, interfaces vary drastically
  (e.g., the older :class:`scipy.integrate.ode` vs. the modern :meth:`scipy.integrate.solve_ivp`),
  and some legacy algorithms do not natively support complex numbers without manual
  splitting into real and imaginary parts.
* **Hardware Limitations**: Standard SciPy integration routines are strictly
  bound to CPU execution, preventing seamless acceleration via GPUs or distributed architectures.
* **Specialized Quantum Algorithms**: Certain high-performance routines tailored
  for quantum mechanics —such as Krylov subspace methods or custom unitary propagators—
  are too domain-specific to be included in general-purpose mathematical libraries.

The :class:`Integrator` class resolves these challenges by abstracting numerical backends
into an interchangeable component, ensuring that switching from a SciPy Fortran solver to
a GPU-accelerated backend requires only changing a configuration string.


Core Design
===========

The abstract base class :class:`Integrator` serves as the common template for all
numerical integration backends.

Deterministic Time-Evolution
----------------------------
For standard solvers, the integrator acts sequentially. Given an initial time $t_0$
and an initial state vector or density matrix, it propagates the state forward to
requested target times. This behaviour is governed by four core methods:

* :meth:`set_state(t, state)`: Configures the initial conditions of the
  integrator, seeding it with the starting time and the state array.
* :meth:`get_state()`: Returns the current internal numerical state of the integration
  loop as a tuple of ``(t, state)``.
* :meth:`integrate(t)`: Steps the numerical solver explicitly to the target time ``t``
  using the internal state configuration.
* :meth:`run(tlist)`: A generator method that yields successive ``(t, state)`` tuples at
  each timestamp specified in the user-provided array ``tlist``.

.. note::
   The output state returned by :meth:`get_state`, :meth:`integrate`, and :meth:`run`
   is consistently packed as a tuple: ``(t, state)``.

.. note::
   All inputs and outputs for states are handled as :class:`qutip.core.data.Data` objects.
   However, an integrator is permitted to change the underlying representation format
   internally during evolution.

   For example, SciPy-based integrators utilize NumPy arrays internally; they will
   automatically convert an incoming initial state from its initial storage type
   to a dense format during setup, and will subsequently return only dense states.

Monte Carlo Trajectories
------------------------
To support Quantum Monte Carlo simulations (:func:`mcsolve`), integrators must provide
hook interfaces capable of handling non-continuous, conditional integration steps:

* :meth:`mcstep(t)`: Governs evolution inside stochastic loops.
  Unlike standard integration, :meth:`mcstep` must safely handle non-increasing time targets
  or backtrack queries (inside the last step range) to assist the parent solver
  in isolating the exact physical moment a quantum collapse event occurs.

For historical reasons, QuTiP delegates the root-finding search for a collapse to the
high-level :class:`MCSolver` loop rather than relying on native ODE "event" triggers.
This design has the advantage of allowing QuTiP to wrap external ODE packages
that lack native event-detection features, guaranteeing consistent behavior across all
ODE providers.


Configuration and State Resetting
=================================

Numerical routines depend on parameters (such as tolerance limits
``atol``/``rtol``, step limits ``nsteps``, etc.) that must be tuned to achieve
optimal efficiency. These parameters are configurable by the developer or end
user via the ``options`` attribute passed down to the integrator instance.
The ``Integrator.options`` attribute operates as a live dictionary containing
the configuration state.

Because used libraries maintain internal state for their ODE solvers,
modifying values within the ``options`` dictionary does not always take effect immediately.
Activating the newly assigned options requires an explicit call to :meth:`reset()`.
This tears down the active backend instance and restarts it with the new options
while preserving the active state and time.


Derivative Format
=================

While ordinary ODE interfaces strictly expect functions to represent the right-hand side (RHS),
quantum equation in QuTiP can profit from alternative formats.
Every integrator exposes an :attr:`RHS_format` class attribute string that dictates
the type of RHS object required during initialization:

* **"callable"**: The integrator expects a function matching the signature ``rhs(t, Data) -> Data``
  or ``rhs(t, Data, out=Data)``. This is the default for most ODE methods.
  For specific Cython-backed ODE implementations, if the passed function is a method of
  :class:`QobjEvo`, it will bypass standard Python overhead and hook directly into the
  Cython bindings of ``QobjEvo.matmul``.
* **"matrix"**: The integrator accepts a complex matrix formatted as a
  :class:`qutip.core.data.Data` object, solving the linear matrix differential equation
  $\frac{dX}{dt} = \text{RHS} \times X$.
* **"solver"**: The integrator takes the parent :class:`Solver` instance that instantiated it
  and builds the RHS internally. This is reserved for specialized integration methods that
  deliberately blend the underlying quantum physics with the numerical method.


Class Attributes Reference
==========================

Every subclass inheriting from :class:`Integrator` must define the following class-level attributes:

.. attribute:: integrator_options

   A class-level dictionary defining the comprehensive set of configuration keys supported
   by the numerical backend along with their default values.
   Upon instantiation, this is copied to an instance-level ``options`` dictionary.

.. attribute:: RHS_format

   A string identifier defining the expected structural format of the Right-Hand Side (RHS)
   derivative function (e.g., ``"callable"``, ``"matrix"``, or ``"solver"``).

.. attribute:: name

   A descriptive, human-readable string representation of the algorithm
   (e.g., ``"SciPy zvode Adams solver"``) used primarily for populating metadata
   inside the final :class:`Result` container.

.. attribute:: method

   A unique short-string identifier key (e.g., ``"adams"`` or ``"bdf"``) registered
   with the solver directory, mapping the option selections directly to the underlying
   class implementation.


Creating New Integrators
========================

To implement a custom numerical backend, create a subclass derived from :class:`Integrator`.
At a minimum, your class must override :meth:`integrate`, :meth:`set_state`, :meth:`get_state`,
and either :meth:`__init__` or :meth:`_prepare`:

.. code-block:: python

    from qutip.solver.integrator import Integrator

    class LinearStepODE(Integrator):
        RHS_format = "callable"
        name = "Simple Linear Step"
        method = "linear"
        # Uses default empty dict if no specialized integrator_options are defined

        def _prepare(self):
            # __init__ automatically prepares the local options dict,
            # stores the rhs in self.derivative, and initializes self._is_set to False.
            # Apply option here!
            # This will be called from __init__ and reset.
            pass

        def set_state(self, t, state):
            self.t = t
            self.state = state
            self._is_set = True

        def integrate(self, t):
            # Basic Euler step example
            self.state = self.state + (t - self.t) * self.derivative(self.t, self.state)
            self.t = t

        def get_state(self):
            return self.t, self.state

The base :class:`Integrator` class automatically provides default implementations for
the :meth:`run` :meth:`reset` and :meth:`mcstep` methods
(though the fallback version of :meth:`mcstep` can be very inefficient).

To expose your custom integrator to a specific physical solver,
register it using the solver class's target method:

.. code-block:: python

    from qutip.solver import SESolver

    SESolver.add_integrator(LinearStepODE, "linear")

Once added, the method is immediately accessible via the high-level functional
wrappers by configuring the options map (e.g., ``options={"method": "linear"}``).
Registering the class with the parent :class:`Solver` base class makes it globally available to all solvers except stochastic solver.

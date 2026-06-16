.. _internal_solveroptions:

Solver Options Infrastructure
=============================

The configuration of QuTiP's physical solvers and numerical integrators is
handled via specialized dictionary objects. Rather than exposing raw Python
dictionaries directly, the framework utilizes an internal subclass,
:class:`_SolverOptions`.


Motivation
==========

Simple dictionary or dataclass lacks the feature needed to

* **Preventing Silent Silent Bugs**: Raw Python dictionaries accept any key-value
  pair without validation. A simple typo, such as typing ``"abstol"`` instead of
  ``"atol"``, would be silently ignored by a standard dictionary, causing the
  underlying integrator to fall back to default values without ever alerting the user.
* **Integrator State Synchronization**: Modern differential equation solvers are
  stateful; they maintain adaptive step histories, interpolation tables, and
  multi-step coefficients. If a user modifies an integration parameter *after*
  a simulation has started stepping forward, the internal state of the underlying
  ODE engine must be forcefully reset to prevent numeric instability or crashes.
  A standard dictionary cannot broadcast its updates to intercept these mid-run changes.
* **Namespace Isolation**: Different integration backends support entirely
  different sets of configuration options. For example, a specialized Krylov subspace
  propagator expects parameters like Krylov dimension limits, which are completely
  meaningless to a standard SciPy ``zvode`` integrator.

The :class:`_SolverOptions` class acts as an intelligent configuration firewall.
By operating as a reactive, validating proxy, it shields developers from managing
manual synchronization loops and protects users from configuration errors, ensuring
that options can safely change dynamically throughout a system's evolutionary life cycle.

Core Features
-------------

The :class:`_SolverOptions` class inherits directly from Python's built-in
:repr:`dict`, behaving like a standard key-value map while injecting four
crucial architectural features:

1. **Reactive State Reset Hooks**
   When an option is updated at runtime, the container calls an internal feedback
   hook. If the system is actively stepping through a simulation, modifying critical
   parameters (such as error tolerances) will automatically trigger an internal
   :meth:`reset` of the numerical integrator backend.

   .. code-block:: python

      solver = SESolver(H, options={"atol": 1e-1})
      solver.start(psi0, 0)
      psi1 = solver.step(1)

      # Dynamically changing a tolerance trigger automatically clears
      # and restarts the active integrator state internally
      solver.options["atol"] = 1e-8
      psi2 = solver.step(2)

2. **Strict Key Validation**
   To prevent silent typos or unsupported flags (e.g., passing ``"abstol"``
   instead of ``"atol"``), the container cross-references all additions and
   mutations against an immutable ``_default`` dictionary. Attempting to inject
   or modify an unknown option raises a :class:`KeyError`.

3. **Default Value Recovery via Erasure**
   Deleting a parameter or setting it to ``None`` restores that parameter back
   to its baseline default configuration seamlessly:

   .. code-block:: python

      # Both actions automatically restore 'atol' back to its default value
      del solver.options["atol"]
      solver.options["atol"] = None

4. **Dynamic Documentation**
   The class instance overrules standard template text by dynamically overriding
   its ``__doc__`` attribute, allowing custom help signatures and parameter
   descriptions to be printed directly for each unique solver setup.

5. **Clean Table Layout Output**
   Printing or inspecting a :class:`_SolverOptions` instance renders listing of
   the dictionary, explicitly labeling which parameters are currently resting on
   their default values.

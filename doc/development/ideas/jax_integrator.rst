JAX integrator
--------------

QuTiP has recently rewritten its solver module to extend its flexibility and
allow different ``Integrator`` (ODE solvers). ``Integrator`` in qutip are the
classes in responsible for the low level computation of the evolution of a
Quantum state represented as a ``Qobj``. These are used in the QuTiP's solvers
(see ``sesolve``, ``mesolve``, ``srmsolve``) to provide a user friendly
interface for the computation of the time dynamics of a wide variety of
systems. QuTiP also includes a data layer to represent its main class,
``Qobj``, using different array interfaces. The data layer has been
successfully extended in previous projects (some of them GSoC) giving birth to
a family of packages ([TensorFlow](https://github.com/qutip/qutip-tensorflow),
[CuPy](https://github.com/qutip/qutip-cupy) and most recently
[Jax](https://github.com/qutip/qutip-jax)) that allow, among other things, GPU
operation and autodifferentiation of QuTiP's ``Qobj``. Those data-layers open
new avenues for quantum control applications in QuTiP, where the
auto-differentiation of a system evolution is expected to improve the
performance of quantum control algorithms. However, preliminary testing of
these data-layers with QuTiPs solvers showed that a data-layer specific
``Integrator`` is needed in order leverage the full potential of
auto-differentiation. Your task as part of the GSoC will be to implement such
``Integrator`` for the qutip-jax data layer, benchamark the result with
existing integrators and provide an example of use of the auto-differentiation
feature, possibly in the context of quantum control.


Project size
============

- 350 hours

Difficulty
==========

- Medium

Deliverables
============

- Implement an Integrator based on jax integrators to allow
- Support JAX `jax.jit` and `jax.grad` for automatic differentiation of
  `Integrator` solutions.
- Benchmark the new integrator with existing alternatives.
- Use Jupyter notebook to demonstrate some simple examples, e.g.: a `jit`
  in the control of a quantum system.

Skills
======

- Familiar with QuTiP and its solvers (feel free to show us some code!)
- Familiar with Python and Git.
- Understand the concepts of Python closures and decorators (which are used
  widely in JAX function transformations).
- Familiar with JAX (beneficial, but not required)


Mentors
=======

- Eric Giguère (eric.giguere@calculquebec.ca)
- Shahnawaz Ahmed (shahnawaz.ahmed95@gmail.com)
- Asier Galicia (agalicia1221@gmail.com)

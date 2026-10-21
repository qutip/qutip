.. _ulme:

**********************************
Universal Lindblad Master Equation
**********************************

.. plot::
      :context: reset
      :include-source: False

      import matplotlib.pyplot as plt
      from scipy import *
      from qutip import *
      import numpy as np

Introduction
============

The :ref:`Bloch-Redfield master equation <bloch_redfield>` derives a system-only master
equation from a microscopic system-bath model, but it does not in general preserve
positivity of the density matrix, except in the secular approximation, where
off-resonant terms are discarded.
The secular approximation is well justified when the system's transition
frequencies are well separated compared to the inverse bath correlation time, but
it can break down for systems with near-degenerate or densely spaced levels, or for
strongly or quickly driven systems.

The Universal Lindblad Master Equation (ULME), introduced by Nathan and Rudner
[Nathan20]_, is a Markovian master equation that is derived under the same
Born-Markov approximation as the Bloch-Redfield equation, but without invoking
the secular approximation.
Instead, it constructs a single jump operator per coupling operator directly
from a frequency-domain filter derived from the bath's power spectrum.
This guarantees, by construction, that the resulting generator is in Lindblad
form, and therefore completely positive and trace preserving (CPTP),
while remaining accurate to the same order in the system-bath coupling as
the Bloch-Redfield equation.
Unlike Bloch-Redfield, the ULME can therefore also be unravelled into quantum
trajectories, e.g. with :func:`.mcsolve`.

The ULME is still a Markovian approximation and is valid under the same conditions
as Bloch-Redfield: weak system-bath coupling and a bath correlation time short
compared to the system’s relaxation time. It makes no assumption about the
system’s spectrum, so it is particularly useful for systems with closely spaced
or degenerate levels, and for driven systems where the secular approximation is
doubtful. In return, building the jump operators is more expensive than the
Bloch-Redfield tensor, especially for time-dependent systems where they are
recomputed during the evolution; when the system is well within the secular
regime, :func:`.brmesolve` is often cheaper and gives similar results.
For strong coupling or long bath memory, a non-Markovian method such as
:ref:`HEOM <heom>` is needed.

Brief Derivation and Definitions
================================

The derivation starts the same way as for the :ref:`Bloch-Redfield equation
<bloch_redfield>`: from the total Hamiltonian :math:`H = H_{\rm S} + H_{\rm B} + H_{\rm I}`
with system-bath interaction :math:`H_{\rm I} = \sum_\alpha X_\alpha \otimes B_\alpha`,
and under the Born-Markov approximation.
We refer to the :ref:`Bloch-Redfield guide <bloch_redfield>` for this part of the
derivation, and the role of the bath correlation function and the power
spectrum :math:`S_\alpha(\omega)` in characterizing each bath; see also the
:ref:`bosonic environments guide <bosonic environments guide>`.

Where the ULME departs from Bloch-Redfield is in how the Markovian generator is
brought into a numerically useful form. Instead of moving to the system's
eigenbasis and discarding non-secular terms, the ULME defines, for each coupling
operator :math:`X_\alpha`, a **jump correlator** (or filter function) :math:`g_\alpha(t)` as

.. math::
    :label: ulme-filter

    g_\alpha(t) = \frac{1}{2\pi} \int_{-\infty}^\infty \mathrm d \omega
    \sqrt{S_\alpha(\omega)} e^{-\mathrm i \omega t}

Because :math:`S_\alpha(\omega)` is non-negative and real, :math:`g_\alpha` satisfies the Hermitian
symmetry :math:`g_\alpha(-t) = g_\alpha(t)^\ast`, and the bath correlation function is
recovered from it as :math:`C_\alpha(t) = \int \mathrm ds\, g_\alpha(-s)\, g_\alpha(t+s)`
-- i.e. :math:`g_\alpha` can be thought of as a kind of "square root" of the bath
correlation function.

The jump operator for coupling operator :math:`X_\alpha` is then built directly from
this jump correlator, by convolving it with the coupling operator in the interaction
picture:

.. math::
    :label: ulme-jump

    L_\alpha(t) = \int_{-\infty}^\infty \mathrm ds\, g_\alpha(s)\, X_{\alpha,I}(t-s) ,

where :math:`X_{\alpha,I}(t-s) = U^\dagger(t-s, t)\, X_\alpha(t-s)\, U(t-s, t)` is the
coupling operator at time :math:`t-s` in the interaction picture referenced to :math:`t`, with
:math:`U` the system propagator. The resulting master equation has the standard
Lindblad form

.. math::
    :label: ulme-master

    \frac{d}{dt}\rho(t) = -i\bigl[H(t) + H_{\rm LS}(t), \rho(t)\bigr] +
    \sum_\alpha \Bigl( L_\alpha(t)\, \rho(t)\, L_\alpha^\dagger(t) -
    \frac12 \bigl\{ L_\alpha^\dagger(t) L_\alpha(t), \rho(t) \bigr\} \Bigr) ,

where :math:`H_{\rm LS}(t)` is a Lamb shift correction, playing the same role as the
energy shift :math:`\lambda_{\alpha\beta}(\omega)` that is neglected in the usual
Bloch-Redfield derivation; here it is kept, and computed from a second-order
integral of the same filter functions. We do not reproduce its (somewhat
lengthy) expression here; see Eq. (29) of [Nathan20]_, or the
implementation, for the details. Including it improves the accuracy of the
equation, but it can be disabled (see below) at the cost of a less accurate
effective Hamiltonian.

For a time-independent Hamiltonian and coupling operator, Eq. :eq:`ulme-jump`
reduces to an explicit expression in the eigenbasis of H. Writing H's
eigenstates as :math:`\left|m\right>` with eigenenergies :math:`E_m`, and
:math:`X_{\alpha,mn} = \left<m\right| X_\alpha \left|n\right>`, the jump operator
becomes

.. math::
    :label: ulme-eigen

    L_\alpha = \sum_{m,n} \sqrt{S_\alpha(E_n - E_m)}\, X_{\alpha,mn}\,
    \left| m \rangle \langle n \right| ,

which only requires diagonalizing H once, rather than solving an ODE for every
jump operator. This is the basis of the ``"eigen"`` construction method
discussed below.

Universal Lindblad Equation in QuTiP
====================================

The ULME is implemented in QuTiP through the function :func:`.ulmesolve` and
the class :class:`.ULMESolver`. They take the system Hamiltonian ``H``, the
initial state, and a list ``a_ops`` of ``(operator, environment)`` pairs --
one per bath -- where each ``operator`` is the (Hermitian)
system-bath coupling operator and each ``environment`` is a
:class:`.BosonicEnvironment` instance (see the
:ref:`bosonic environments guide <bosonic environments guide>`).
Unlike :func:`.brmesolve`'s ``a_ops``, bare spectral functions and fermionic baths
are not supported here.

In simple cases it can be used in place of :func:`.brmesolve`.

.. code-block:: python

    result = ulmesolve(H, psi0, tlist, a_ops=[(X, env)], e_ops=e_ops)

Internally, :func:`.ulmesolve` builds a Lamb-shifted Hamiltonian and a set of
jump operators and then evolves the system with :func:`.mesolve`, so it accepts
the same ``e_ops``, ``args`` and general options as :func:`.mesolve`,
in addition to the ULME-specific options below.

As an example, let's look at a 5-level system with near-degenerate energies:

.. plot::
    :context: close-figs

    N = 5
    X = (destroy(N) + create(N)) * 0.2

    H = qdiags([-1, 0, 0, 0.01, 1])
    a_ops = (X, OhmicEnvironment(T=0.0, alpha=0.5, wc=10, s=1.))
    psi0 = basis(N, N-1)
    e_ops = [num(N)]
    tlist = np.linspace(0, 5, 101)

    result_ul = ulmesolve(H, psi0, tlist, a_ops=[a_ops], e_ops=e_ops, options={"use_lamb_shift": False, "store_states": True})
    result_ul_ls = ulmesolve(H, psi0, tlist, a_ops=[a_ops], e_ops=e_ops, options={"store_states": True})
    result_br = brmesolve(H, psi0, tlist, a_ops=[a_ops], e_ops=e_ops, sec_cutoff=-1, options={"store_states": True})

    plt.plot(result_ul.times, result_ul.expect[0], label="universal lindblad")
    plt.plot(result_ul_ls.times, result_ul_ls.expect[0], label="universal lindblad with lamb shift")
    plt.plot(result_br.times, result_br.expect[0], label="bloch-redfield")
    plt.ylabel("<n>")
    plt.xlabel("time")
    plt.legend()
    plt.show()


    min_ul = [min(state.eigenenergies()) for state in result_ul.states]
    min_ls = [min(state.eigenenergies()) for state in result_ul_ls.states]
    min_br = [min(state.eigenenergies()) for state in result_br.states]
    plt.plot(result_ul.times, min_ul, label="universal lindblad")
    plt.plot(result_ul_ls.times, min_ls, label="universal lindblad with lamb shift")
    plt.plot(result_br.times, min_br, label="bloch-redfield")
    plt.title("smallest eigenvalue")
    plt.xlabel("time")
    plt.legend()
    plt.show()


In this regime, we see all three evolution shows the same feature,
but with :func:`.brmesolve` being quite similar to :func:`.ulmesolve` without Lamb shift.
We also observe that the state becomes unphysical (negative eigenvalue) with Bloch-Redfield,
but stays physical with ULME.


For a more complex example, we can look at a driven qubit with multiple dissipation channels:

.. plot::
  :context: close-figs

  H = sigmaz() + sigmax() * coefficient(lambda t: 2. * np.sin(t * 2 * np.pi))
  a_ops = [
      (sigmax(), UnderDampedEnvironment(T=0.3, lam=0.1, w0=1.2, gamma=0.8)),
      (sigmay(), UnderDampedEnvironment(T=4.0, lam=0.7, w0=0.5, gamma=0.5)),
  ]

  psi0 = basis(2, 0)
  tlist = np.linspace(0, 1.0, 101)
  result = ulmesolve(
      H, psi0, tlist, a_ops, e_ops={"x": sigmax(), "y": sigmay(), "z": sigmaz()},
      options={"use_lamb_shift": False}
  )
  for key in result.e_data:
      plt.plot(result.times, result.e_data[key], label="ULME: " + key)

  result = brmesolve(
      H, psi0, tlist, a_ops, e_ops={"x": sigmax(), "y": sigmay(), "z": sigmaz()},
      sec_cutoff=-1,
  )
  for key in result.e_data:
      plt.plot(result.times, result.e_data[key], ":", label="BRME: " + key)

  plt.ylabel("expectation value")
  plt.xlabel("time")
  plt.legend()


Here again, we see that :func:`.ulmesolve` gives results similar to :func:`.brmesolve`.


The ``UL_transform`` function
-----------------------------

As with :func:`.bloch_redfield_tensor`, it is sometimes useful to obtain the
Lamb-shifted Hamiltonian and jump operators without running the evolution, for
example to inspect them, or pass them to other solvers
(:func:`.mcsolve`, :func:.`steadystate`, ...).
The function :func:`.UL_transform` does this:

.. code-block:: python

    H_eff, c_ops = UL_transform(H, a_ops=[(X, env)])
    result = mesolve(H_eff, psi0, tlist, c_ops=c_ops, e_ops=e_ops)

which, together with :func:`.mesolve`, is equivalent to calling
:func:`.ulmesolve` directly.


.. plot::
    :context: close-figs

    N = 8
    sz = sigmaz() & qeye(N)
    sx = sigmax() & qeye(N)
    a = qeye(2) & destroy(N)

    H = sz + a.dag() * a + 0.2 * sx * (a + a.dag())
    a_op = ((a + a.dag()) * 0.3, OhmicEnvironment(T=1.0, alpha=0.8, wc=0.4, s=1.))
    psi0 = tensor(basis(2, 0), coherent(N, 2)).unit()
    e_ops = [a.dag() @ a]
    tlist = np.linspace(0, 5, 101)

    H_shifted, c_ops = UL_transform(H, a_ops=[a_op])

    result_me = mesolve(H_shifted, psi0, tlist, c_ops=c_ops, e_ops=e_ops)
    result_mc = mcsolve(
        H_shifted, psi0, tlist, c_ops=c_ops, e_ops=e_ops,
        ntraj=20, options={"keep_runs_results": True}
    )

    plt.plot(result_me.times, result_me.expect[0], label="ME")
    plt.plot(result_mc.times, result_mc.runs_expect[0].T, "--", lw=0.5)
    plt.plot(result_mc.times, result_mc.average_expect[0], "-.", label="MC")
    plt.ylabel("cavity population")
    plt.xlabel("time")
    plt.legend()


Options
-------

The main ULME-specific options, set through the ``options`` dictionary as with
other solvers, are:

- ``use_lamb_shift`` (default ``True``): whether to compute and include the
  Lamb shift :math:`H_{\rm LS}` in Eq. :eq:`ulme-master`.
- ``ULME_creation`` (default ``"propagator"``): the construction method, as
  described above.

A number of further options control the numerical details of each
construction method (the integration limits for the ``"eigen"`` method's Lamb
shift, the ODE integrator and convergence tolerance for the ``"propagator"``
method, and the options used to build the jump correlator itself). The full
list, with defaults, is documented in :obj:`ULMESolver.options`.


Construction methods: eigenbasis vs. propagator
-----------------------------------------------

The jump operators and Lamb shift can be constructed in two ways, selected
with the ``ULME_creation`` option:

- ``"eigen"`` diagonalizes H and builds the jump operator directly from
  Eq. :eq:`ulme-eigen`, using the bath's power spectrum. Only applicable to a
  time-independent Hamiltonian and coupling operator, but more stable than the
  other method.

- ``"propagator"`` (the default) evaluates the convolution of Eq.
  :eq:`ulme-jump`. This works for time-dependent Hamiltonians and coupling
  operators as well, and is generally the faster option when the Lamb shift is included.
  However it can be unstable and not work with every bath. This method does the
  integration of the convolution centered on the jump correlator up to a cutoff
  determined dynamically from the jump correlator. For time dependent systems,
  this must be computed at each internal step of the evolution. The option
  ``conv_tol`` can be set to stop the integration when it's detected to be
  converged which can speed up the computation. But this detection is basic
  and may result in increased numerical error.

Both methods are available so that one can be used to validate the other;
for most use cases, the default ``"propagator"`` method is the right choice.

Defining a bath from its jump correlator
----------------------------------------

With the ``"propagator"`` method, the only property of the bath that is used
is its jump correlator :math:`g(t)` (Eq. :eq:`ulme-filter`). By default it is
computed numerically from the power spectrum, by Fourier transforming
:math:`\sqrt{S(\omega)}` and interpolating the result. No pre-defined environment
has an analytical jump correlator. If the Fourier transform of
:math:`\sqrt{S(\omega)}` is known in closed form, or when :math:`g(t)` comes
from another source, giving it directly avoids the error of the numerical estimate.
:meth:`.BosonicEnvironment.from_jump_correlator` builds an environment from
either a function or an array:


.. code-block:: python

    # from a function
    def my_jump_correlator(t):
        ...

    env = BosonicEnvironment.from_jump_correlator(my_jump_correlator, tMax=tMax)

    # or, sampled on a grid
    env = BosonicEnvironment.from_jump_correlator(jc_array, tlist=tlist)


The jump correlator must satisfy two conditions:

- the Hermitian symmetry :math:`g(-t) = g(t)^\ast`;
- it must have decayed to a negligible value by ``tMax`` (or the end of
  ``tlist``). ``tMax`` is also how far the solver integrates, so a value
  that is too small silently truncates the jump operator.

A sampled jump correlator is used for :math:`t \ge 0` only, extended to :math:`t \lt 0` by the
Hermitian symmetry, and taken as zero beyond the end of tlist. tlist must start at 0.


Numerical accuracy
==================

Checking the jump correlator
----------------------------

The jump correlator :math:`g_\alpha(t)` of Eq. :eq:`ulme-filter` can be a delicate
function to evaluate numerically -- for some environments it decays slowly, or
has structure concentrated close to t=0 -- so before running a potentially
long simulation it is worth checking that it has been resolved sensibly for
the chosen environment and parameters.
Every :class:`.BosonicEnvironment` can build this function directly, as a
fast, spline-interpolated callable, independently of :func:`.ulmesolve`:

.. code-block:: python

    g = env.jump_correlator_function(
        atol=1e-4, rtol=1e-3, wMax=None, wMin=None, tMax=None
    )

This is the same callable used internally by the ``"propagator"`` construction
method (and accepts the same keyword arguments as the ``JC_options`` solver
option), so inspecting it this way -- in particular checking that it has
decayed to a negligible value by ``g.tMax`` -- is a good sanity check on those
options before calling :func:`.ulmesolve`.  The
:class:`.DrudeLorentzEnvironment` is particularly slow to converge and needs
attention.

.. plot::
    :context: close-figs

    env = DrudeLorentzEnvironment(T=0.1, lam=10., gamma=1.)
    # Obtain default ``JC_options`` dict
    JC_options = ULMESolver.solver_options["JC_options"].copy()

    jc = env.jump_correlator_function(**JC_options)
    t_Max = jc.tMax

    # BosonicEnvironment can be built from jump_correlator obtained with
    # jump_correlator_function, fixing the options: tMax, etc.
    env_from_jc = BosonicEnvironment.from_jump_correlator(jc)

    tlist = np.linspace(-0.1, 1.2*t_Max, 2001)

    plt.plot(tlist, jc(tlist).real, "k", label="real")
    plt.plot(tlist, jc(tlist).imag, "k:", label="imag")
    plt.title("Jump correlator")
    plt.show()

    # These should match for the jump_correlator to be valid.
    tlist = np.linspace(-0.1, 1.2*t_Max, 2001)
    # Correlator function and power spectra defined analytically
    # for pre-defined environment.
    plt.plot(tlist, env.correlation_function(tlist).real, "b", label="Analytical (real)")
    plt.plot(tlist, env.correlation_function(tlist).imag, "b:", label="Analytical (imag)")

    # When a bath is defined from its jump correlator
    # The correlator function is obtained by computing the
    # jump correlator convolution
    plt.plot(tlist, env_from_jc.correlation_function(tlist).real, "r", label="Reconstructed (real)")
    plt.plot(tlist, env_from_jc.correlation_function(tlist).imag, "r:", label="Reconstructed (imag)")
    plt.legend()
    plt.title("Correlation function")
    plt.show()


Here we can see the cutoff is too tight and the analytical
correlation function is not negligible past the integration range and a larger
``tMax`` should be set in ``JC_options``.


Truncation of the Lamb shift integral (``"eigen"`` method)
----------------------------------------------------------

In the ``"eigen"`` method, the jump operator only needs the power spectrum at
the Bohr frequencies, so it is not affected by this truncation. The Lamb
shift, however, is a principal value integral over frequency of the form
:math:`\int \mathrm d \omega\, \sqrt{S(\omega - e_1)}\sqrt{S(\omega + e_2)} / \omega`.
With :math:`e_i` difference in energy level of the Hamiltonian.
It is evaluated numerically on the finite interval
:math:`[-\Lambda, \Lambda]`, with :math:`\Lambda` set by the option
``eigen_pv_integral_limits`` (default ``30``, in the units of ``H``).
The calculation therefore assumes that :math:`S(\omega)` is effectively zero
outside of this window, and anything beyond it is silently dropped. Take
particular care when

- the bath cutoff, the temperature or the system energies are not small
  compared to ``30``;
- the spectrum decays slowly (e.g. a Drude-Lorentz bath).

To check, increase the limit and see if the Lamb shift changes:

.. code-block:: python

    for lim in [1, 5, 15, 30, 60]:
        H_eff, _ = UL_transform(
            H, (X, env),
            options={"ULME_creation": "eigen", "eigen_pv_integral_limits": lim},
        )
        print(lim, (H_eff(0) - H).norm())

This option has no effect when ``use_lamb_shift=False``. The Lamb shift
integral is also the main cost of the ``"eigen"`` method, which grows quickly
with the Hilbert space dimension (one numerical integral per distinct pair of
Bohr frequencies).


.. plot::
    :context: reset
    :include-source: false
    :nofigs:

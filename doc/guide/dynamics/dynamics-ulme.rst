.. _ulme:

Universal Lindblad Master Equation
==================================

.. plot::
      :context: reset
      :include-source: False

      import matplotlib.pyplot as plt
      from scipy import *
      from qutip import *
      import numpy as np

Introduction
------------

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
form, and therefore completely positive and trace preserving (CPTP) at every
order of truncation, while remaining accurate to the same order in the
system-bath coupling as the Bloch-Redfield equation.
Unlike Bloch-Redfield, the ULME can therefore also be unravelled into quantum
trajectories, e.g. with :func:`.mcsolve`.

The ULME is still a Markovian approximation: it is accurate when the bath
correlation time is short compared to the system's relaxation time, with an
error that can be bounded rigorously in terms of this ratio. It does not
assume anything about the system's spectrum, so it is particularly useful for
systems with closely-spaced or degenerate levels, and for driven or strongly
coupled systems where the secular approximation is doubtful.

Brief Derivation and Definitions
--------------------------------

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
    :label: ule-filter

    g_\alpha(t) = \frac{1}{2\pi} \int_{-\infty}^\infty \mathrm d \omega
    \sqrt{S_\alpha(\omega)} e^{-\mathrm i \omega t}

Because :math:`S_\alpha(\omega)` is positive and real, :math:`g_\alpha` satisfies the Hermitian
symmetry :math:`g_\alpha(-t) = g_\alpha(t)^\ast`, and the bath correlation function is
recovered from it as :math:`C_\alpha(t) = \int \mathrm ds\, g_\alpha(-s)\, g_\alpha(t+s)`
-- i.e. :math:`g_\alpha` can be thought of as a kind of "square root" of the bath
correlation function.

The jump operator for coupling operator :math
:`X_\alpha` is then built directly from
this jump correlator, by convolving it with the coupling operator in the interaction
picture:

.. math::
    :label: ule-jump

    L_\alpha(t) = \int_{-\infty}^\infty \mathrm ds\, g_\alpha(s)\, X_{\alpha,
    I}(t-s) ,

where :math:`X_{\alpha,I}(t-s) = U^\dagger(t, t-s)\, X_\alpha\, U(t, t-s)` is the
coupling operator at time :math:`t-s` in the interaction picture referenced to :math:`t`, with
:math:`U` the system propagator. The resulting master equation has the standard
Lindblad form

.. math::
    :label: ule-master

    \frac{d}{dt}\rho(t) = -i\bigl[H(t) + H_{\rm LS}(t), \rho(t)\bigr] +
    \sum_\alpha \Bigl( L_\alpha(t)\, \rho(t)\, L_\alpha^\dagger(t) -
    \frac12 \bigl\{ L_\alpha^\dagger(t) L_\alpha(t), \rho(t) \bigr\} \Bigr) ,

where :math:`H_{\rm LS}(t)` is a Lamb shift correction, playing the same role as the
energy shift :math:`\lambda_{\alpha\beta}(\omega)` that is neglected in the usual
Bloch-Redfield derivation; here it is kept, and computed from a second-order
integral of the same filter functions. We do not reproduce its (somewhat
lengthy) expression here; see Eq. (22)-(24) of [Nathan20]_, or the
implementation, for the details. Including it improves the accuracy of the
equation, but it can be disabled (see below) at the cost of a less accurate
effective Hamiltonian.

For a time-independent Hamiltonian and coupling operator, Eq. :eq:`ule-jump`
reduces to an explicit expression in the eigenbasis of H. Writing H's
eigenstates as :math:`\left|m\right>` with eigenenergies E_m, and
:math:`X_{\alpha,mn} = \left<m\right| X_\alpha \left|n\right>`, the jump operator
becomes

.. math::
    :label: ule-eigen

    L_\alpha = 2\pi \sum_{m,n} S_\alpha(\omega)(E_n - E_m)**0.5\, X_{\alpha,mn}\,
    \left|m\right>\left<n\right| ,

which only requires diagonalizing H once, rather than solving an ODE for every
jump operator. This is the basis of the `"eigen"` construction method
discussed below.

Universal Lindblad Equation in QuTiP
------------------------------------

The ULME is implemented in QuTiP through the function :func:`.ulmesolve` and
the class :class:`.ULMESolver`. They take the system Hamiltonian ``H``, the
initial state, and a list ``a_ops`` of ``(operator, environment)`` pairs --
one per independent bath -- where each ``operator`` is the (Hermitian)
system-bath coupling operator and each ``environment`` is a
:class:`.BosonicEnvironment` instance (see the
:ref:`bosonic environments guide <bosonic environments guide>`).
Unlike :func:`.brmesolve`'s ``a_ops``, bare spectral functions are not supported here.

.. code-block:: python

    result = ulmesolve(H, psi0, tlist, a_ops=[(X, env)], e_ops=e_ops)

Internally, :func:`.ulmesolve` builds a Lamb-shifted Hamiltonian and a set of
jump operators -- one per bath -- and then evolves the system with
:func:`.mesolve`, so it accepts the same ``e_ops``, ``args`` and general
options as :func:`.mesolve`, in addition to the ULME-specific options below.


As an example, consider a driven qubit dispersively coupled to a damped
cavity. We compare the ULME, with and without the Lamb shift correction,
against the Bloch-Redfield equation:

.. plot::
    :context:

    N = 5
    sz = sigmaz() & qeye(N)
    sx = sigmax() & qeye(N)
    a = qeye(2) & destroy(N)

    H = sz + a.dag() * a + 0.1 * sx * (a + a.dag())
    a_ops = (a + a.dag(), OhmicEnvironment(T=1.0, alpha=0.8, wc=0.4, s=1.))
    psi0 = tensor(basis(2, 0), coherent(N, 2)).unit()
    e_ops = [a @ a.dag()]
    tlist = np.linspace(0, 5, 101)

    result_ul = ulmesolve(H, psi0, tlist, a_ops=[a_ops], e_ops=e_ops, options={"use_lamb_shift": False})
    result_ul_ls = ulmesolve(H, psi0, tlist, a_ops=[a_ops], e_ops=e_ops)
    result_br = brmesolve(H, psi0, tlist, a_ops=[a_ops], e_ops=e_ops)

    plt.plot(result_ul.times, result_ul.expect[0], label="universal lindblad")
    plt.plot(result_ul_ls.times, result_ul_ls.expect[0], label="universal lindblad with lamb shift")
    plt.plot(result_br.times, result_br.expect[0], label="bloch-redfield")
    plt.legend()


The ``UL_transform`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with :func:`.bloch_redfield_tensor`, it is sometimes useful to obtain the
Lamb-shifted Hamiltonian and jump operators without running the evolution, for
example to inspect them, or to combine them with additional, purely
phenomenological, collapse operators. The function :func:`.UL_transform` does
this:

.. code-block:: python

    H_eff, c_ops = UL_transform(H, a_ops=[(X, env)])
    result = mesolve(H_eff, psi0, tlist, c_ops=c_ops, e_ops=e_ops)

which, together with :func:`.mesolve`, is equivalent to calling
:func:`.ulmesolve` directly, and the returned ``H_eff``, ``c_ops`` can equally
be passed to :func:`.mcsolve` for a trajectory unravelling.


.. plot::
    :context:

    H_shifted, dissipator = UL_transform(H, a_ops=[a_ops])

    result_me = mesolve(H_shifted, psi0, tlist, c_ops=dissipator, e_ops=e_ops)
    result_mc = mcsolve(H_shifted, psi0, tlist, c_ops=dissipator, e_ops=e_ops, ntraj=20, options={"keep_runs_results": True})

    plt.plot(result_me.times, result_me.expect[0])
    plt.plot(result_mc.times, result_mc.runs_expect[0].T, "--", lw=0.5)
    plt.plot(result_mc.times, result_mc.average_expect[0], "-.")



Construction methods: eigenbasis vs. propagator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The jump operators and Lamb shift can be constructed in two ways, selected
with the ``ULME_creation`` option:

- ``"eigen"`` diagonalizes H and builds the jump operator directly from
  Eq. :eq:`ule-eigen`, using the bath's power spectrum. Only applicable to a
  time-independent Hamiltonian and coupling operator, but more stable than the
  other method.

- ``"propagator"`` (the default) evaluates the convolution of Eq.
  :eq:`ule-jump`. This works for time-dependent Hamiltonians and coupling
  operators as well, and is generally the faster option even for time-independent
  systems, especially when the Lamb shift is included. However it can be unstable
  and not work with every bath.

Both methods are available so that one can be used to validate the other;
for most use cases, the default ``"propagator"`` method is the right choice.

Analytical jump correlator
^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to create an environment from a jump correlator, either as an array or a function.
The passed function is expected to satisfy the Hermitian symmetry
:math:`g(-t) = g(t)^\ast` and to decay to effectively zero by ``tMax``


.. code-block:: python

    def my_jump_correlator(t):
        ...

    env = BosonicEnvironment.from_jump_correlator(my_jump_correlator, tMax=1.0)

    # or

    env = BosonicEnvironment.from_jump_correlator(jc_array, tlist=tlist)

Options
^^^^^^^

The main ULME-specific options, set through the ``options`` dictionary as with
other solvers, are:

- ``use_lamb_shift`` (default ``True``): whether to compute and include the
  Lamb shift H_{\rm LS} in Eq. :eq:`ule-master`.
- ``ULME_creation`` (default ``"propagator"``): the construction method, as
  described above.

A number of further options control the numerical details of each
construction method (the integration limits for the ``"eigen"`` method's Lamb
shift, the ODE integrator and convergence tolerance for the ``"propagator"``
method, and the options used to build the jump correlator itself). The full
list, with defaults, is documented in :obj:`ULMESolver.options`.

Checking the jump correlator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The jump correlator :math:`g_\alpha(t)` of Eq. :eq:`ule-filter` can be a delicate
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
    :context:


    env = DrudeLorentzEnvironment(T=0.1, lam=10., gamma=1.)
    # Obtain default options
    JC_options = ULMESolver.solver_options["JC_options"].copy()

    jc = env.jump_correlator_function(**JC_options)
    t_Max = jc.tMax

    # BosonicEnvironment can be build from jump_correlator obtained with
    # jump_correlator_function, fixing the options: tMax, etc.
    env_from_jc = BosonicEnvironment.from_jump_correlator(jc)

    tlist = np.linspace(-0.1, 1.2*t_Max, 2001)

    plt.plot(tlist, jc(tlist).real)
    plt.plot(tlist, jc(tlist).imag)
    plt.show()

    # These should match for the jump_correlator to be valid.
    tlist = np.linspace(-0.1, 1.2*t_Max, 2001)
    # Correlator function and power spectra definited analytically
    # for pre-defined environment.
    plt.plot(tlist, env.correlation_function(tlist).real, "b")
    plt.plot(tlist, env.correlation_function(tlist).imag, "b:")

    # When a bath is definied from it's jump correlator
    # It's correlator function is obtained from the it's convolution
    plt.plot(tlist, env_from_jc.correlation_function(tlist).real, "r")
    plt.plot(tlist, env_from_jc.correlation_function(tlist).imag, "r:")
    plt.show()




.. plot::
    :context: reset
    :include-source: false
    :nofigs:

####################
Bosonic Environments
####################

In this section we consider a simple two-level system coupled to a
Drude-Lorentz bosonic bath. The system Hamiltonian, :math:`H_{sys}`, and the bath
spectral density, :math:`J_D`, are

.. math::

    H_{sys} &= \frac{\epsilon \sigma_z}{2} + \frac{\Delta \sigma_x}{2}

    J_D &= \frac{2\lambda \gamma \omega}{(\gamma^2 + \omega^2)},

We will demonstrate how to describe the bath using two different expansions
of the spectral density correlation function (Matsubara's expansion and
a Padé expansion), how to evolve the system in time and how to calculate
the steady state.

First we will do this in the simplest way, using the built-in implementations of
the two bath expansions, :class:`DrudeLorentzBath` and
:class:`DrudeLorentzPadeBath`.

Afterwards, we will show how to calculate the bath expansion coefficients and to
use those coefficients to construct your own bath description so that you can
implement your own bosonic baths.

A notebook containing a complete example similar to this one implemented in
BoFiN can be found in
`example notebook 1a <https://github.com/tehruhn/bofin/blob/main/examples/example-1a-Spin-bath-model-basic.ipynb>`__).


Describing the system and bath
------------------------------

First, let us construct the system Hamiltonian, :math:`H_{sys}`, and the initial
system state, ``rho0``:

.. plot::
    :context: reset
    :nofigs:

    from qutip import basis, sigmax, sigmaz

    # The system Hamiltonian:
    eps = 0.5  # energy of the 2-level system
    Del = 1.0  # tunnelling term
    H_sys = 0.5 * eps * sigmaz() + 0.5 * Del* sigmax()

    # Initial state of the system:
    rho0 = basis(2,0) * basis(2,0).dag()

Now let us describe the bath properties:

.. plot::
    :context:
    :nofigs:

    # Bath properties:
    gamma = 0.5  # cut off frequency
    lam = 0.1  # coupling strength
    T = 0.5  # temperature

    # System-bath coupling operator:
    Q = sigmaz()

where :math:`\gamma` (``gamma``), :math:`\lambda` (``lam``) and :math:`T` are
the parameters of a Drude-Lorentz bath, and ``Q`` is the coupling operator
between the system and the bath.

We may the pass these parameters to either ``DrudeLorentzBath`` or
``DrudeLorentzPadeBath`` to construct an expansion of the bath correlations:

.. plot::
    :context:
    :nofigs:

    from qutip.nonmarkov.heom import DrudeLorentzBath
    from qutip.nonmarkov.heom import DrudeLorentzPadeBath

    # Number of expansion terms to retain:
    Nk = 2

    # Matsubara expansion:
    bath = DrudeLorentzBath(Q, lam, T, Nk, gamma)

    # Padé expansion:
    bath = DrudeLorentzPadeBath(Q, lam, T, Nk, gamma)

Where ``Nk`` is the number of terms to retain within the expansion of the
bath.


System and bath dynamics
------------------------

Now we are ready to construct a solver:

.. plot::
    :context:
    :nofigs:

    from qutip.nonmarkov.heom import HEOMSolver
    from qutip import Options

    max_depth = 5  # maximum hierarchy depth to retain
    options = Options(nsteps=15_000)

    solver = HEOMSolver(H_sys, bath, max_depth=max_depth, options=options)

and to calculate the system evolution as a function of time:

.. code-block:: python

    tlist = [0, 10, 20]  # times to evaluate the system state at
    result = solver.run(rho0, tlist)

The ``max_depth`` parameter determines how many levels of the hierarchy to
retain. As a first approximation hierarchy depth may be thought of as similar
to the order of Feynman Diagrams (both classify terms by increasing number
of interactions).

The ``result`` is a standard QuTiP results object with the attributes:

- ``times``: the times at which the state was evaluated (i.e. ``tlist``)
- ``states``: the system states at each time
- ``expect``: the values of each ``e_ops`` at each time
- ``ado_states``: see below

If ``ado_return=True`` is passed to ``.run(...)`` the full set of auxilliary
density operators (ADOs) that make up the hierarchy at each time will be
returned as ``.ado_states``. We will describe how to use these to determine
other properties, such as system-bath currents, later in the guide
(see :ref:`heom-ado-states`).

If one has a full set of ADOs from a previous call of ``.run(...)`` you may
supply it as the initial state of the solver by calling
``.run(result.ado_states[-1], tlist, ado_init=True)``.

As with other QuTiP solvers, if expectation operators or functions are supplied
using ``.run(..., e_ops=[...])`` the expectation values are available in
``result.expect``.

Below we run the solver again, but use ``e_ops`` to store the expectation
values of the population of the system states and the coherence:

.. plot::
    :context:

    # Define the operators that measure the populations of the two
    # system states:
    P11p = basis(2,0) * basis(2,0).dag()
    P22p = basis(2,1) * basis(2,1).dag()

    # Define the operator that measures the 0, 1 element of density matrix
    # (corresonding to coherence):
    P12p = basis(2,0) * basis(2,1).dag()

    # Run the solver:
    tlist = np.linspace(0, 20, 101)
    result = solver.run(rho0, tlist, e_ops={"11": P11p, "22": P22p, "12": P12p})

    # Plot the results:
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(result.times, result.expect["11"], 'b', linewidth=2, label="P11")
    axes.plot(result.times, result.expect["12"], 'r', linewidth=2, label="P12")
    axes.set_xlabel(r't', fontsize=28)
    axes.legend(loc=0, fontsize=12)


Steady-state
------------

Using the same solver, we can also determine the steady state of the
combined system and bath using:

.. plot::
    :context:
    :nofigs:

    steady_state, steady_ados = solver.steady_state()

where ``steady_state`` is the steady state of the system and ``steady_ados``
if the steady state of the full hierarchy. The ADO states are
described more fully in :ref:`heom-ado-states`.


Matsubara Terminator
--------------------

When constructing the Drude-Lorentz bath we have truncated the expansion at
``Nk = 2`` terms and ignore the remaining terms.

However, since the coupling to these higher order terms is comparatively weak,
we may consider the interaction with them to be Markovian, and construct an
additional Lindbladian term that captures their interaction with the system and
the lower order terms in the expansion.

This additional term is called the ``terminator`` because it terminates the
expansion.

The :class:`DrudeLorentzBath` and :class:`DrudeLorentzPadeBath` both provide
a means of calculating the terminator for a given expansion:

.. plot::
    :context:
    :nofigs:

    # Matsubara expansion:
    bath = DrudeLorentzBath(Q, lam, T, Nk, gamma, terminator=True)

    # Padé expansion:
    bath = DrudeLorentzPadeBath(Q, lam, T, Nk, gamma, terminator=True)

    # Add terminator to the system Liouvillian:
    HL = liouvillian(H_sys) + bath.terminator

    # Construct solver:
    solver = HEOMSolver(HL, bath, max_depth=max_depth, options=options)

This captures the Markovian effect of the remaining terms in the expansion
without having to fully model many more terms.


Matsubara expansion coefficients
--------------------------------

So far we have relied on the built-in :class:`DrudeLorentzBath` to construct
the Drude-Lorentz bath expansion for us. Now we will calculate the coefficients
ourselves and construct a :class:`BosonicBath` directly. A similar procedure
can be used to apply :class:`HEOMSolver` to any bosonic bath for which we
can calculate the expansion coefficients.

The real and imaginary parts of the correlation function, :math:`C(t)`, for the
bosonic bath is expanded in an expontential series:

.. math::

      C(t) &= C_{real}(t) + i C_{imag}(t)

      C_{real}(t) &= \sum_{k=0}^{\infty} c_{k,real} e^{- \nu_{k,real} t}

      C_{imag}(t) &= \sum_{k=0}^{\infty} c_{k,imag} e^{- \nu_{k,imag} t}

In the specific case of Matsubara expansion for the Drude-Lorentz bath, the
coefficients of this expansion are, for the real part, :math:`C_{real}(t)`:

.. math::

    \nu_{k,real} &= \begin{cases}
        \gamma                & k = 0\\
        {2 \pi k} / {\beta }  & k \geq 1\\
    \end{cases}

    c_{k,real} &= \begin{cases}
        \lambda \gamma [\cot(\beta \gamma / 2) - i]             & k = 0\\
        \frac{4 \lambda \gamma \nu_k }{ (\nu_k^2 - \gamma^2)\beta}    & k \geq 1\\
    \end{cases}

and the imaginary part, :math:`C_{imag}(t)`:

.. math::

    \nu_{k,imag} &= \begin{cases}
        \gamma                & k = 0\\
        0                     & k \geq 1\\
    \end{cases}

    c_{k,imag} &= \begin{cases}
        - \lambda \gamma      & k = 0\\
        0                     & k \geq 1\\
    \end{cases}

And now the same numbers calculated in Python:

.. plot::
    :context:
    :nofigs:

    # Convenience functions and parameters:

    def cot(x):
        return 1. / np.tan(x)

    beta = 1. / T

    # Number of expansion terms to calculate:
    Nk = 2

    # C_real expansion terms:
    ck_real = [lam * gamma / np.tan(gamma / (2 * T))]
    ck_real.extend([
        (8 * lam * gamma * T * np.pi * k * T /
            ((2 * np.pi * k * T)**2 - gamma**2))
        for k in range(1, Nk + 1)
    ])
    vk_real = [gamma]
    vk_real.extend([2 * np.pi * k * T for k in range(1, Nk + 1)])

    # C_imag expansion terms (this is the full expansion):
    ck_imag = [lam * gamma * (-1.0)]
    vk_imag = [gamma]

After all that, constructing the bath is very straight forward:

.. plot::
    :context:
    :nofigs:

    bath = BosonicBath(Q, ck_real, vk_real, ck_imag, vk_imag)

And we're done!

The :class:`BosonicBath` can be used with the :class:`HEOMSolver` in exactly
the same way as the baths we constructed previously using the built-in
Drude-Lorentz bath expansions.


Multiple baths
--------------

.. todo::

    Clean up this section to use the new multiple baths feature.


The above example describes a single environment parameterized by the lists of
coefficients and frequencies in the correlation functions.

For multiple environments, the list of coupling operators and bath properties
must all be extended in a particular way.  Note this functionality differs in
the case of the Fermionic solver.

For the Bosonic solver, for ``N`` baths, each ``ckAR``, ``vkAR``, ``ckAI``, and
``vkAI`` are extended ``N`` times with the appropriate number of terms of that
bath.

On the other hand, the list of coupling operators is defined in such a way that
the terms corresponding to the real cooefficients are **given first**, and the
imaginary terms after. Thus if each bath has :math:`N_k` coefficients, the list
of coupling operators is of length :math:`N_k \times (N_R + N_I)`.

This is best illustrated by the example in `example notebook 2
<https://github.com/tehruhn/bofin/blob/main/examples/example-2-FMO-example.ipynb>`_.
In that case each bath is identical, and there are seven baths, each with a
unique coupling operator defined by a projector onto a single state:

.. code-block:: python

    ckAR = [pref * lam * gamma * (cot(gamma / (2 * T))) + 0.j]
    ckAR.extend([(pref * 4 * lam * gamma * T *  2 * np.pi * k * T / (( 2 * np.pi * k * T)**2 - gamma**2))+0.j for k in range(1,Nk+1)])
    vkAR = [gamma+0.j]
    vkAR.extend([2 * np.pi * k * T + 0.j for k in range(1,Nk+1)])
    ckAI = [pref * lam * gamma * (-1.0) + 0.j]
    vkAI = [gamma+0.j]

    NR = len(ckAR)
    NI = len(ckAI)
    Q2 = []
    ckAR2 = []
    ckAI2 = []
    vkAR2 = []
    vkAI2 = []
    for m in range(7):
        Q2.extend([ basis(7,m)*basis(7,m).dag() for kk in range(NR)])
        ckAR2.extend(ckAR)
        vkAR2.extend(vkAR)

    for m in range(7):
        Q2.extend([ basis(7,m)*basis(7,m).dag() for kk in range(NI)])
        ckAI2.extend(ckAI)
        vkAI2.extend(vkAI)

.. plot::
    :context: reset
    :nofigs:

    # reset the context at the end

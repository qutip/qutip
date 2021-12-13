.. _stochastic:

*******************************************
Stochastic Solver
*******************************************

.. _stochastic-intro:

Homodyne detection
==================
Homodyne detection is an extension of the photocurrent method where the output
is mixed with a strong external source allowing to get information about the
phase of the system. With this method, the resulting detection rate depends is

.. math::
   :label: jump_rate

   \tau = \tr \left((\gamma^2 + \gamma (C+C^\dagger) + C^\dagger C)\rho \right)

With :math:`\gamma`, the strength of the external beam and :math:`C` the collapse
operator. When the beam is very strong :math:`(\gamma >> C^\dagger C)`,
the rate becomes a constant term plus a term proportional to the quadrature of
the system.

Closed system
-------------
.. Stochastic Schrodinger equation

In closed systems, the resulting stochastic differential equation is

.. math::
    :label: jump_ssesolve

    d \psi(t) = - i H \psi(t) dt
                     - \sum_n \left( \frac{C_n^\dagger C_n}{2} -\frac{e_n}{2} C_n
                     + \frac{e_n^2}{8} \right) \psi(t) dt
                     + \sum_n \left( C_n - \frac{e_n}{2} \right) \psi(t) dW

with

.. math::
   :label: jump_matrix_element

   e_n = \left<\psi(t)|C_n + C_n^\dagger|\psi(t)\right>

Here :math:`dW` is a Wiener increment with :math:`E[dW] = dt`.

In QuTiP, you can solve this equation using :func:`qutip.ssesolve` passing the argument `method="homodyne"`. See the example below.

.. plot::
    :context:

    import qutip

    kappa = 0.1
    g = 0.25
    times = np.linspace(0.0, 10.0, 201)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = (2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm
         + 2*np.pi*g*(sm*a.dag() + sm.dag()*a))
    data = ssesolve(H, psi0, times, sc_ops=[np.sqrt(kappa) * a],
                    e_ops=[a.dag()*a, sm.dag()*sm], method="homodyne")

    fig, ax = plt.subplots()
    ax.plot(times, data.expect[0], times, data.expect[1])
    ax.set_title('Homodyne Detection - Closed System')
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.legend(("cavity photon number", "atom excitation probability"))

For more examples, see this `notebook <https://nbviewer.ipython.org/github/qutip/qutip-notebooks/blob/master/development/development-ssesolve-tests.ipynb>`_.

Open system
--------------
.. Stochastic Master equation

In open systems, 2 types of collapse operators are considered, :math:`S_i`
represent the dissipation in the environment, :math:`C_i` are monitored operators.
The deterministic part of the evolution, described by the Liouvillian :math:`L`, takes into account all operators :math:`S_i` and :math:`C_i`:

.. math::
    :label: liouvillian

    L(\rho(t)) = - i[H(t),\rho(t)]
                 + \sum_n D[S_n] \rho
                 + \sum_i D[C_i], \rho,

where

.. math::
    :label: dissipator

    D[A, \rho] = \frac{1}{2} \left[2 A \rho C^\dagger
               - \rho A^\dagger A - A^\dagger A \rho \right].

The stochastic evolution is given solely by the operators :math:`C_i`

.. math::
    :label: stochastic_smesolve

    d_{2,i} = C_i \rho(t) + \rho(t) C_i^\dagger - \tr \left(C_i \rho (t)
                     + \rho(t) C_i^\dagger \right)\rho(t),

resulting in the stochastic differential equation

.. math::
    :label: sde_smesolve

    d \rho(t) = L(\rho(t)) d t + \sum_i d_{2,i}  dW

In QuTiP, the solver :func:`qutip.smesolve` is used to solve the equation above. As in the closed system case, homodyne detection can be simulated by choosing `method="homodyne"`, as in the example below.

.. plot::
    :context: close-figs

    kappa = 0.1
    g = 0.25
    times = np.linspace(0.0, 10.0, 201)
    rho0 = tensor(fock_dm(2, 0), fock_dm(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*g*(sm*a.dag() + sm.dag()*a)
    data = smesolve(H, rho0, times, sc_ops=[np.sqrt(kappa) * a],
                    e_ops=[a.dag()*a, sm.dag()*sm], method="homodyne", nsubsteps=100)

    fig, ax = plt.subplots()
    ax.plot(times, data.expect[0], times, data.expect[1])
    ax.set_title('Homodyne Detection - Open System')
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.legend(("cavity photon number", "atom excitation probability"))

For more examples, see the following `notebook <https://nbviewer.ipython.org/github/qutip/qutip-notebooks/blob/master/development/development-smesolve-tests.ipynb>`_.

..

To save the measurement results of the stochastic simulations, one should set
the argument `store_measurement=True` and provide the appropriate `dW_factors`.
For the system above, we can have

.. plot::
    :context: close-figs

    kappa = 0.1
    g = 0.25
    times = np.linspace(0.0, 10.0, 201)
    rho0 = tensor(fock_dm(2, 0), fock_dm(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*g*(sm*a.dag() + sm.dag()*a)
    data = smesolve(H, rho0, times, sc_ops=[np.sqrt(kappa) * a],
                    e_ops=[a.dag()*a, sm.dag()*sm], method="homodyne",
                    store_measurement=True, dW_factors=[1], nsubsteps=100)

    fig, ax = plt.subplots()
    ax.plot(times, np.array(data.measurement).mean(axis=0)[:,0].real/np.sqrt(0.1), times, data.expect[1])
    ax.set_title('Measurement of X Quadrature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.legend(("cavity photon number", "atom excitation probability"))

Heterodyne detection
--------------------
With heterodyne detection, two measurements are made in order to obtain
information about 2 orthogonal quadratures at once. Similar to homodyne case, it can be simulated by passing the argument `method="heterodyne"` for both :func:`qutip.ssesolve` and :func:`qutip.smesolve`.

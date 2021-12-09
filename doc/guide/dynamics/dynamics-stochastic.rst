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

   \tau = \tr \left((\gamma^2 + \gamma (C+C^\dag) + C^\dag C)\rho \right)

With :math:`\gamma`, the strength of the external beam and :math:`C` the collapse
operator. When the beam is very strong :math:`(\gamma >> C^\dag C)`,
the rate becomes a constant term plus a term proportional to the quadrature of
the system.

Closed system
-------------
.. Stochastic Schrodinger equation

In closed systems, the resulting stochastic differential equation is

.. math::
	:label: jump_ssesolve

	\delta \psi(t) = - i H \psi(t) \delta t
	                 - \sum_n \left( \frac{C_n^{+} C_n}{2} -\frac{e_n}{2} C_n
					 + \frac{e_n^2}{8} \right) \psi \delta t
	                 + \sum_n \left( C_n - \frac{e_n}{2} \right) \psi \delta \omega

with

.. math::
   :label: jump_matrix_element

   e_n = \left<\psi(t)|C_n + C_n^{+}|\psi(t)\right>

Here :math:`\delta \omega` is a Wiener increment.

In QuTiP, this is available with the function :func:`ssesolve`.

.. plot::
    :context:

    times = np.linspace(0.0, 10.0, 201)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    data = ssesolve(H, psi0, times, sc_ops=[np.sqrt(0.1) * a], e_ops=[a.dag()*a, sm.dag()*sm], method="homodyne")

    plt.figure()
    plt.plot(times, data.expect[0], times, data.expect[1])
    plt.title('Homodyne time evolution')
    plt.xlabel('Time')
    plt.ylabel('Expectation values')
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()


Open system
--------------
.. Stochastic Master equation

In open systems, 2 types of collapse operators are considered, :math:`S_i`
represent the dissipation in the environment, :math:`C_i` are monitored operators.
The deterministic part of the evolution is the liouvillian with both types of
collapses

.. math::
	:label: liouvillian

	L(\rho(t)) = - i[H(t),\rho(t)]
	             + \sum_n D(S_n, \rho)
				 + \sum_i D(C_i, \rho),

with

.. math::
 	:label: disipator

	D(C, \rho) = \frac{1}{2} \left[2 C \rho(t) C^{+}
			   - \rho(t) C^{+} C - C^{+} C \rho(t) \right].

The stochastic part is given by

.. math::
	:label: stochastic_smesolve

	d_2 = \left(C \rho(t) + \rho(t) C^{+} - \rm{tr}\left(C \times \rho
					 + \rho \times C^{+} \right)\rho(t) \right),

resulting in the stochastic differential equation

.. math::
	:label: sde_smesolve

	\delta \rho(t) = L(\rho(t)) \delta t + d_2  \delta \omega

The function :func:`smesolve` covert these cases in QuTiP.

Heterodyne detection
--------------------
With heterodyne detection, two measurements are made in order to obtain
information about 2 orthogonal quadratures at once.

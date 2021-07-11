.. _stochastic_photo:

********************************
Stochastic Solver - Photocurrent
********************************

.. _photocurrent-intro:

Photocurrent method, like monte-carlo method, allows for simulating an
individual realization of the system evolution under continuous measurement.

Closed system
-------------

.. photocurent_Schrodinger_equation

Photocurrent evolution have the state evolve deterministically between quantum jumps.
During the deterministic part, the system evolve by schrodinger equation with a
non-hermitian, norm conserving effective Hamiltonian.

.. math::
	:label: pssesolve_heff

	H_{\rm eff}=H_{\rm sys}+
	\frac{i\hbar}{2}\left( -\sum_{n}C^{+}_{n}C_{n}+ |C_{n} \psi |^2\right).

With :math:`C_{n}`, the collapse operators.
This effective Hamiltonian is equivalent to the monte-carlo effective
Hamiltonian with an extra term to keep the state normalized.
At each time step of :math:`\delta t`, the wave function has a probability

.. math::
	:label: pssesolve_jump_prob

	\delta p_{n} = \left<\psi(t)|C_{n}^{+}C_{n}|\psi(t)\right>  \delta t

of making a quantum jump. :math:`\delta t` must be chosen small enough to keep
that probability small :math:`\delta p << 1`. *If multiple jumps happen at the
same time step, the state become unphysical.*
Each jump result in a sharp variation of the state by,

.. math::
	:label: pssesolve_jump

	\delta \psi = \left( \frac{C_n \psi} {\left| C_n \psi  \right|} - \psi \right)

The basic photocurrent method directly integrates these equations to the first-order.
Starting from a state :math:`\left|\psi(0)\right>`, it evolves the state according to

.. math::
	:label: pssesolve_sde

	\delta \psi(t) = - i H_{\rm sys} \psi(t) \delta t + \sum_n \left(
					 -\frac{C_n^{+} C_n}{2}  \psi(t) \delta t
					 + \frac{ \left| C_n \psi  \right| ^2}{2} \delta t
	                 +  \delta N_n \left( \frac{C_n \psi}
					 {\left| C_n \psi  \right|} - \psi \right)\right),

for each time-step.
Here :math:`\delta N = 1` with a probability of :math:`\delta \omega` and
:math:`\delta N_n = 0` with a probability of :math:`1-\delta \omega`.

Trajectories obtained with this algorithm are equivalent to those obtained with
monte-carlo evolution (up to :math:`O(\delta t^2)`).
In most cases, :func:`qutip.mcsolve` is more efficient than
:func:`qutip.photocurrent_sesolve`.

Open system
-----------
.. photocurent_Master_equation

Photocurrent approach allows to obtain trajectories for a system with
both measured and dissipative interaction with the bath.
The system evolves according to the master equation between jumps with a modified
liouvillian

.. math::
	:label: master_equation

	L_{\rm eff}(\rho(t)) = L_{\rm sys}(\rho(t)) +
	                      \sum_{n}\left(
						  \rm{tr} \left(C_{n}^{+}C_{n}  \rho C_{n}^{+}C_{n} \right)
						  - C_{n}^{+}C_{n}  \rho C_{n}^{+}C_{n} \right),

with the probability of jumps in a time step :math:`\delta t` given by

.. math::
	:label: psmesolve_rate

	\delta p = \rm{tr} \left( C \rho C^{+} \right) \delta t.

After a jump, the density matrix become

.. math::

	\rho' = \frac{C \rho C^{+}}{\rm{tr} \left( C \rho C^{+} \right)}.

The evolution of the system at each time step if thus given by

.. math::
	:label: psmesolve_sde

	\rho(t + \delta t) = \rho(t) + L_{\rm eff}(\rho) \delta t + \delta N
	\left(\frac{C \rho C^{+}}{\rm{tr} \left( C \rho C^{+} \right)} - \rho \right).

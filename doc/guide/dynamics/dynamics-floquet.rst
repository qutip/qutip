.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _floquet:

*****************
Floquet Formalism
*****************

.. _floquet-intro:

Introduction
============

Many time-dependent problems of interest are periodic. The dynamics of such systems can be solved for directly by numerical integration of the Schrödinger or Master equation, using the time-dependent Hamiltonian. But they can also be transformed into time-independent problems using the Floquet formalism. Time-independent problems can be solve much more efficiently, so such a transformation is often very desirable. 

In the standard derivations of the Lindblad and Bloch-Redfield master equations the Hamiltonian describing the system under consideration is assumed to be time independent. Thus, strictly speaking, the standard forms of these master equation formalisms should not blindly be applied to system with time-dependent Hamiltonians. However, in many relevant cases, in particular for weak driving, the standard master equations still turns out to be useful for many time-dependent problems. But a more rigorous approach would be to rederive the master equation taking the time-dependent nature of the Hamiltonian into account from the start. The Floquet-Markov Master equation is one such a formalism, with important applications for strongly driven systems [see e.g., Grifoni et al., Physics Reports 304, 299 (1998)].

Here we give an overview of how the Floquet and Floquet-Markov formalisms can be used for solving time-dependent problems in QuTiP. To introduce the terminology and naming conventions used in QuTiP we first give a brief summary of quantum Floquet theory.

.. _floquet-unitary:

Floquet theory for unitary evolution
====================================

The Schrödinger equation with a time-dependent Hamiltonian :math:`H(t)` is

.. math::
   :label: eq_td_schrodinger

	H(t)\Psi(t) = i\hbar\frac{\partial}{\partial t}\Psi(t),

where :math:`\Psi(t)` is the wave function solution. Here we are interested in problems with periodic time-dependence, i.e., the Hamiltonian satisfies :math:`H(t) = H(t+T)` where :math:`T` is the period. According to the Floquet theorem, there exist solutions to :eq:`eq_td_schrodinger` on the form 

.. math::
   :label: eq_floquet_states
   
    \Psi_\alpha(t) = \exp(-i\epsilon_\alpha t/\hbar)\Phi_\alpha(t),

where :math:`\Psi_\alpha(t)` are the *Floquet states* (i.e., the set of wave function solutions to the Schrödinger equation), :math:`\Phi_\alpha(t)=\Phi_\alpha(t+T)` are the periodic *Floquet modes*, and :math:`\epsilon_\alpha` are the *quasienergy levels*. The quasienergy levels are constants in time, but only uniquely defined up to multiples of :math:`2\pi/T` (i.e., unique value in the interval :math:`[0, 2\pi/T]`).

If we know the Floquet modes (for :math:`t \in [0,T]`) and the quasienergies for a particular :math:`H(t)`, we can easily decompose any initial wavefunction :math:`\Psi(t=0)` in the Floquet states and immediately obtain the solution for arbitrary :math:`t`

.. math::
   :label: eq_floquet_wavefunction_expansion
   
    \Psi(t) = \sum_\alpha c_\alpha \Psi_\alpha(t) = \sum_\alpha c_\alpha \exp(-i\epsilon_\alpha t/\hbar)\Phi_\alpha(t),

where the coefficients :math:`c_\alpha` are determined by the initial wavefunction :math:`\Psi(0) = \sum_\alpha c_\alpha \Psi_\alpha(0)`.

This formalism is useful for finding :math:`\Psi(t)` for a given :math:`H(t)` only if we can obtain the Floquet modes :math:`\Phi_a(t)` and quasienergies :math:`\epsilon_\alpha` more easily than directly solving :eq:`eq_td_schrodinger`. By substituting :eq:`eq_floquet_states` into the Schrödinger equation :eq:`eq_td_schrodinger` we obtain an eigenvalue equation for the Floquet modes and quasienergies

.. math::
   :label: eq_floquet_eigen_problem
   
    \mathcal{H}(t)\Phi_\alpha(t) = \epsilon_\alpha\Phi_\alpha(t),
    
where :math:`\mathcal{H}(t) = H(t) - i\hbar\partial_t`. This eigenvalue problem could be solved analytically or numerically, but in QuTiP we use an alternative approach for numerically finding the Floquet states and quasienergies [see e.g. Creffield et al., Phys. Rev. B 67, 165301 (2003)]. Consider the propagator for the time-dependent Schrödinger equation :eq:`eq_td_schrodinger`, which by definition satisfies
    
.. math::

    U(T+t,t)\Psi(t) = \Psi(T+t).

Inserting the Floquet states from :eq:`eq_floquet_states` into this expression results in 

.. math::
    U(T+t,t)\exp(-i\epsilon_\alpha t/\hbar)\Phi_\alpha(t) = \exp(-i\epsilon_\alpha(T+t)/\hbar)\Phi_\alpha(T+t),

or, since :math:`\Phi_\alpha(T+t)=\Phi_\alpha(t)`,

.. math::
    U(T+t,t)\Phi_\alpha(t) = \exp(-i\epsilon_\alpha T/\hbar)\Phi_\alpha(t) = \eta_\alpha \Phi_\alpha(t),

which shows that the Floquet modes are eigenstates of the one-period propagator. We can therefore find the Floquet modes and quasienergies :math:`\epsilon_\alpha = -\hbar\arg(\eta_\alpha)/T` by numerically calculating :math:`U(T+t,t)` and diagonalizing it. In particular this method is useful to find :math:`\Phi_\alpha(0)` by calculating and diagonalize :math:`U(T,0)`. 

The Floquet modes at arbitrary time :math:`t` can then be found by propagating :math:`\Phi_\alpha(0)` to :math:`\Phi_\alpha(t)` using the wave function propagator :math:`U(t,0)\Psi_\alpha(0) = \Psi_\alpha(t)`, which for the Floquet modes yields

.. math::

    U(t,0)\Phi_\alpha(0) = \exp(-i\epsilon_\alpha t/\hbar)\Phi_\alpha(t),

so that :math:`\Phi_\alpha(t) = \exp(i\epsilon_\alpha t/\hbar) U(t,0)\Phi_\alpha(0)`. Since :math:`\Phi_\alpha(t)` is periodic we only need to evaluate it for :math:`t \in [0, T]`, and from :math:`\Phi_\alpha(t \in [0,T])` we can directly evaluate :math:`\Phi_\alpha(t)`, :math:`\Psi_\alpha(t)` and :math:`\Psi(t)` for arbitrary large :math:`t`.

Floquet formalism in QuTiP
--------------------------

QuTiP provides a family of functions to calculate the Floquet modes and quasi energies, Floquet state decomposition, etc., given a time-dependent Hamiltonian on the *callback format*, *list-string format* and *list-callback format* (see, e.g., :func:`qutip.mesolve` for details). 

Consider for example the case of a strongly driven two-level atom, described by the Hamiltonian

.. math::
   :label: eq_driven_qubit

    H(t) = -\frac{1}{2}\Delta\sigma_x - \frac{1}{2}\epsilon_0\sigma_z + \frac{1}{2}A\sin(\omega t)\sigma_z.
    
In QuTiP we can define this Hamiltonian as follows

>>> delta = 0.2 * 2*pi; eps0 = 1.0 * 2*pi; A = 2.5 * 2*pi; omega = 1.0 * 2*pi
>>> H0 = - delta/2.0 * sigmax() - eps0/2.0 * sigmaz()
>>> H1 = A/2.0 * sigmaz()
>>> args = {'w': omega}
>>> H = [H0, [H1, 'sin(w * t)']]

The :math:`t=0` Floquet modes corresponding to the Hamiltonian :eq:`eq_driven_qubit` can then be calculated using the :func:`qutip.floquet.floquet_modes` function, which returns lists containing the Floquet modes and the quasienergies

>>> T = 2*pi / omega
>>> f_modes, f_energies = floquet_modes(H, T, args)
>>> f_energies
array([ 2.83131211, -2.83131211])
>>> f_modes0
[Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data =
[[ 0.39993745+0.554682j]
 [ 0.72964232+0.j      ]],
 Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data =
[[ 0.72964232+0.j      ]
 [-0.39993745+0.554682j]]]

For some problems interesting observations can be draw from the quasienergy levels alone. Consider for example the quasienergies for the driven two-level system introduced above as a function of the driving amplitude, calculated and plotted in the following example. For certain driving amplitudes the quasienergy levels cross. Since the the quasienergies can be associated with the time-scale of the long-term dynamics due that the driving, degenerate quasienergies indicates a "freezing" of the dynamics (sometimes known as coherent destruction of tunneling).

.. plot:: guide/scripts/floquet_ex0.py
   :width: 4.0in
   :include-source:	 

Given the Floquet modes at :math:`t=0`, we obtain the Floquet mode at some later time :math:`t` using the function :func:`qutip.floquet.floquet_mode_t`: 

>>> f_modes_t = floquet_modes_t(f_modes_0, f_energies, 2.5, H, T, args)
>>> f_modes_t
[Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data =
[[-0.03189259+0.6830849j ]
 [-0.61110159+0.39866357j]],
 Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data =
[[-0.61110159-0.39866357j]
 [ 0.03189259+0.6830849j ]]]

The purpose of calculating the Floquet modes is to find the wavefunction solution to the original problem :eq:`eq_driven_qubit` given some initial state :math:`\left|\psi_0\right>`. To do that, we first need to decompose the initial state in the Floquet states, using the function :func:`qutip.floquet.floquet_state_decomposition`

>>> psi0 = rand_ket(2)
>>> f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
[(0.81334464307183041-0.15802444453870021j),
 (-0.17549465805005662-0.53169576969399113j)]

and given this decomposition of the initial state in the Floquet states we can easily evaluate the wavefunction that is the solution to :eq:`eq_driven_qubit` at an arbitrary time :math:`t` using the function :func:`qutip.floquet.floquet_wavefunction_t`

>>> t = 10 * rand()
>>> psi_t = floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, args)  
>>> psi_t
Quantum object: dims = [[2], [1]], shape = [2, 1], type = ket
Qobj data =
[[-0.29352582+0.84431304j]
 [ 0.30515868+0.32841589j]]

The following example illustrates how to use the functions introduced above to calculate and plot the time-evolution of :eq:`eq_driven_qubit`.

.. plot:: guide/scripts/floquet_ex1.py
   :width: 4.0in
   :include-source:	 

Pre-computing the Floquet modes for one period
----------------------------------------------

When evaluating the Floquet states or the wavefunction at many points in time it is useful to pre-compute the Floquet modes for the first period of the driving with the required resolution. In QuTiP the function :func:`qutip.floquet.floquet_modes_table` calculates a table of Floquet modes which later can be used together with the function :func:`qutip.floquet.floquet_modes_t_lookup` to efficiently lookup the Floquet mode at an arbitrary time. The following example illustrates how the example from the previous section can be solved more efficiently using these functions for pre-computing the Floquet modes.

.. plot:: guide/scripts/floquet_ex2.py
   :width: 4.0in
   :include-source:	 

Note that the parameters and the Hamiltonian used in this example is not the same as in the previous section, and hence the different appearance of the resulting figure.

.. _floquet-unitary:


Floquet theory for dissipative evolution
========================================

A driven system that is interacting with its environment is not necessarily well described by the standard Lindblad master equation, since its dissipation process could be time-dependent due to the driving. In such cases a rigorious approach would be to take the driving into account when deriving the master equation. This can be done in many different ways, but one way common approach is to derive the master equation in the Floquet basis. That approach results in the so-called Floquet-Markov master equation, see Grifoni et al., Physics Reports 304, 299 (1998) for details. 


The Floquet-Markov master equation in QuTiP
-------------------------------------------

The QuTiP function :func:`qutip.floquet.fmmesolve` implements the Floquet-Markov master equation. It calculates the dynamics of a system given its initial state, a time-dependent hamiltonian, a list of operators through which the system couples to its environment and a list of corresponding spectral density functions that describes the environment. In contrast to the :func:`qutip.mesolve` and :func:`qutip.mcsolve`, and the :func:`qutip.floquet.fmmesolve` does characterize the environment with dissipation rates, but extract the strength of the coupling to the environment from the noise spectral-density functions and the instantaneous Hamiltonian parameters (similar to the Bloch-Redfield master equation solver :func:`qutip.bloch_redfield.brmesolve`). 

.. note::

    Currently the :func:`qutip.floquet.fmmesolve` can only accept a single environment coupling operator and spectral density function.

The noise spectral density function of the environment is implemented as a python callback function that is passed to the solver. For example:

>>> gamma1 = 0.1
>>> def noise_spectrum(omega):
>>>     return 0.5 * gamma1 * omega/(2*pi)

The other parameters are similar to the :func:`qutip.mesolve` and :func:`qutip.mcsolve`, and the same format for the return value is used :class:`qutip.Odedata`. The following example extends the example studied above, and uses :func:`qutip.floquet.fmmesolve` to introduce dissipation into the calculation

.. plot:: guide/scripts/floquet_ex3.py
   :width: 4.0in
   :include-source:	 


.. _monte:

*******************************************
Monte Carlo Solver
*******************************************


.. _monte-intro:

Introduction
=============

Where as the density matrix formalism describes the ensemble average over many identical realizations of a quantum system, the Monte Carlo (MC), or quantum-jump approach to wave function evolution, allows for simulating an individual realization of the system dynamics.  Here, the environment is continuously monitored, resulting in a series of quantum jumps in the system wave function, conditioned on the increase in information gained about the state of the system via the environmental measurements.  In general, this evolution is governed by the Schrödinger equation with a **non-Hermitian** effective Hamiltonian

.. math::
	:label: heff

	H_{\rm eff}=H_{\rm sys}-\frac{i\hbar}{2}\sum_{i}C^{+}_{n}C_{n},

where again, the :math:`C_{n}` are collapse operators, each corresponding to a separate irreversible process with rate :math:`\gamma_{n}`.  Here, the strictly negative non-Hermitian portion of Eq. :eq:`heff` gives rise to a reduction in the norm of the wave function, that to first-order in a small time :math:`\delta t`, is given by :math:`\left<\psi(t+\delta t)|\psi(t+\delta t)\right>=1-\delta p` where

.. math::
	:label: jump

	\delta p =\delta t \sum_{n}\left<\psi(t)|C^{+}_{n}C_{n}|\psi(t)\right>,

and :math:`\delta t` is such that :math:`\delta p \ll 1`.  With a probability of remaining in the state :math:`\left|\psi(t+\delta t)\right>` given by :math:`1-\delta p`, the corresponding quantum jump probability is thus Eq. :eq:`jump`.  If the environmental measurements register a quantum jump, say via the emission of a photon into the environment, or a change in the spin of a quantum dot, the wave function undergoes a jump into a state defined by projecting :math:`\left|\psi(t)\right>` using the collapse operator :math:`C_{n}` corresponding to the measurement

.. math::
	:label: project

	\left|\psi(t+\delta t)\right>=C_{n}\left|\psi(t)\right>/\left<\psi(t)|C_{n}^{+}C_{n}|\psi(t)\right>^{1/2}.

If more than a single collapse operator is present in Eq. :eq:`heff`, the probability of collapse due to the :math:`i\mathrm{th}`-operator :math:`C_{i}` is given by

.. math::
	:label: pcn

	P_{i}(t)=\left<\psi(t)|C_{i}^{+}C_{i}|\psi(t)\right>/\delta p.

Evaluating the MC evolution to first-order in time is quite tedious.  Instead, QuTiP uses the following algorithm to simulate a single realization of a quantum system.  Starting from a pure state :math:`\left|\psi(0)\right>`:

- **Ia:** Choose a random number :math:`r_1` between zero and one, representing the probability that a quantum jump occurs.

- **Ib:** Choose a random number :math:`r_2` between zero and one, used to select which collapse operator was responsible for the jump.

- **II:** Integrate the Schrödinger equation, using the effective Hamiltonian :eq:`heff` until a time :math:`\tau` such that the norm of the wave function satisfies :math:`\left<\psi(\tau)\right.\left|\psi(\tau)\right> = r_1`, at which point a jump occurs.

- **III:** The resultant jump projects the system at time :math:`\tau` into one of the renormalized states given by Eq. :eq:`project`.  The corresponding collapse operator :math:`C_{n}` is chosen such that :math:`n` is the smallest integer satisfying:

.. math::
    :label: mc3

    \sum_{i=1}^{n} P_{n}(\tau) \ge r_2

where the individual :math:`P_{n}` are given by Eq. :eq:`pcn`.  Note that the left hand side of Eq. :eq:`mc3` is, by definition, normalized to unity.

- **IV:** Using the renormalized state from step III as the new initial condition at time :math:`\tau`, draw a new random number, and repeat the above procedure until the final simulation time is reached.


.. _monte-qutip:

Monte Carlo in QuTiP
====================

In QuTiP, Monte Carlo evolution is implemented with the :func:`qutip.mcsolve` function. It takes nearly the same arguments as the :func:`qutip.mesolve`
function for master-equation evolution, except that the initial state must be a ket vector, as oppose to a density matrix, and there is an optional keyword parameter ``ntraj`` that defines the number of stochastic trajectories to be simulated.  By default, ``ntraj=500`` indicating that 500 Monte Carlo trajectories will be performed.

To illustrate the use of the Monte Carlo evolution of quantum systems in QuTiP, let's again consider the case of a two-level atom coupled to a leaky cavity. The only differences to the master-equation treatment is that in this case we invoke the :func:`qutip.mcsolve` function instead of :func:`qutip.mesolve`

.. plot::
    :context: reset

    from qutip.solver.mcsolve import MCSolver, mcsolve

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 8))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))
    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], e_ops=[a.dag() * a, sm.dag() * sm])

    plt.figure()
    plt.plot(times, data.expect[0], times, data.expect[1])
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time')
    plt.ylabel('Expectation values')
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()

.. guide-dynamics-mc1:

The advantage of the Monte Carlo method over the master equation approach is that only the state vector is required to be kept in the computers memory, as opposed to the entire density matrix. For large quantum system this becomes a significant advantage, and the Monte Carlo solver is therefore generally recommended for such systems. For example, simulating a Heisenberg spin-chain consisting of 10 spins with random parameters and initial states takes almost 7 times longer using the master equation rather than Monte Carlo approach with the default number of trajectories running on a quad-CPU machine.  Furthermore, it takes about 7 times the memory as well. However, for small systems, the added overhead of averaging a large number of stochastic trajectories to obtain the open system dynamics, as well as starting the multiprocessing functionality, outweighs the benefit of the minor (in this case) memory saving. Master equation methods are therefore generally more efficient when Hilbert space sizes are on the order of a couple of hundred states or smaller.


Monte Carlo Solver Result
-------------------------

The Monte Carlo solver returns a :class:`qutip.MultitrajResult` object consisting of expectation values and/or states.
The main difference with :func:`qutip.mesolve`'s :class:`qutip.Result` is that it optionally stores the result of each trajectory together with their averages.
When trajectories are stored, ``result.runs_expect`` is a list over the expectation operators, trajectories and times in that order.
The averages are stored in ``result.average_expect`` and the standard derivation of the expectation values in ``result.std_expect``.
When the states are returned, ``result.runs_states`` will be an array of length ``ntraj``. Each element contains an array of "Qobj" type ket with the same number of elements as ``times``. ``result.average_states`` is a list of density matrices computed as the average of the states at each time step.
Furthermore, the output will also contain a list of times at which the collapse occurred, and which collapse operators did the collapse. These can be obtained in  ``result.col_times`` and ``result.col_which`` respectively.


Photocurrent
------------

The photocurrent, previously computed using the ``photocurrent_sesolve`` and ``photocurrent_sesolve`` functions, are now included in the output of :func:`qutip.solver.mcsolve` as ``result.photocurrent``.


.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 8))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))
    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], e_ops=[a.dag() * a, sm.dag() * sm])

    plt.figure()
    plt.plot((times[:-1] + times[1:])/2, data.photocurrent[0])
    plt.title('Monte Carlo Photocurrent')
    plt.xlabel('Time')
    plt.ylabel('Photon detections')
    plt.show()


.. _monte-ntraj:

Changing the Number of Trajectories
-----------------------------------

By default, the ``mcsolve`` function runs 500 trajectories.
This value was chosen because it gives good accuracy, Monte Carlo errors scale as :math:`1/n` where :math:`n` is the number of trajectories, and simultaneously does not take an excessive amount of time to run.
However, you can change the number of trajectories to fit your needs.
In order to run 1000 trajectories in the above example, we can simply modify the call to ``mcsolve`` like:

.. plot::
    :context: close-figs

    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1000)

where we have added the keyword argument ``ntraj=1000`` at the end of the inputs.
Now, the Monte Carlo solver will calculate expectation values for both operators, ``a.dag() * a, sm.dag() * sm`` averaging over 1000 trajectories.



.. _monte-reuse:

Reusing Hamiltonian Data
------------------------

.. note:: This section covers a specialized topic and may be skipped if you are new to QuTiP.


In order to solve a given simulation as fast as possible, the solvers in QuTiP take the given input operators and break them down into simpler components before passing them on to the ODE solvers.
Although these operations are reasonably fast, the time spent organizing data can become appreciable when repeatedly solving a system over, for example, many different initial conditions.
In cases such as this, the Monte Carlo Solver may be reused after the initial configuration, thus speeding up calculations.


Using the previous example, we will calculate the dynamics for two different initial states, with the Hamiltonian data being reused on the second call

.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])
    data1 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=100)
    psi1 = tensor(fock(2, 0), coherent(10, 2 - 1j))
    data2 = solver.run(psi1, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=100)

    plt.figure()
    plt.plot(times, data1.expect[0], "b", times, data1.expect[1], "r", lw=2)
    plt.plot(times, data2.expect[0], 'b--', times, data2.expect[1], 'r--', lw=2)
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Expectation values', fontsize=14)
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()

.. guide-dynamics-mc2:

The ``MCSolver`` also allows adding new trajectories after the first computation. This is shown in the next example where the results of two separated runs with identical conditions are merged into a single ``result`` object.

.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])
    data1 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1, seed=1)
    data2 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1, seed=3)
    data_merged = data1 + data2

    plt.figure()
    plt.plot(times, data1.expect[0], times, data1.expect[1], lw=2)
    plt.plot(times, data2.expect[0], '--', times, data2.expect[1], '--', lw=2)
    plt.plot(times, data_merged.expect[0], ':', times, data_merged.expect[1], ':', lw=2)
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Expectation values', fontsize=14)
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()


This can be used to explore the convergence of the Monte Carlo solver.
For example, the following code block plots expectation values for 1, 10 and 100 trajectories:

.. plot::
    :context: close-figs

    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])

    data1 = solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=1)
    data10 = data1 + solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=9)
    data100 = data10 + solver.run(psi0, times, e_ops=[a.dag() * a, sm.dag() * sm], ntraj=90)

    expt1 = data1.expect
    expt10 = data10.expect
    expt100 = data100.expect

    plt.figure()
    plt.plot(times, expt1[0], label="ntraj=1")
    plt.plot(times, expt10[0], label="ntraj=10")
    plt.plot(times, expt100[0], label="ntraj=100")
    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time')
    plt.ylabel('Expectation values')
    plt.legend()
    plt.show()


.. openmcsolve:

Open Systems
------------

``mcsolve`` can be used to study system with have measured and dissipative interaction with the bath.
This is done by using a liouvillian including the dissipative interaction instead of an Hamiltonian.

.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 8))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))
    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    L = liouvillian(H, [0.01 * sm, np.sqrt(0.1) * a])
    data = mcsolve(L, psi0, times, [np.sqrt(0.1) * a], e_ops=[a.dag() * a, sm.dag() * sm])

    plt.figure()
    plt.plot((times[:-1] + times[1:])/2, data.photocurrent[0])
    plt.title('Monte Carlo Photocurrent')
    plt.xlabel('Time')
    plt.ylabel('Photon detections')
    plt.show()


.. plot::
    :context: reset
    :include-source: false
    :nofigs:

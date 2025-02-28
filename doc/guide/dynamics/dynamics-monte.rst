.. _monte:

*******************************************
Monte Carlo Solver
*******************************************


.. _monte-intro:

Introduction
============

Where as the density matrix formalism describes the ensemble average over many
identical realizations of a quantum system, the Monte Carlo (MC), or
quantum-jump approach to wave function evolution, allows for simulating an
individual realization of the system dynamics.  Here, the environment is
continuously monitored, resulting in a series of quantum jumps in the system
wave function, conditioned on the increase in information gained about the
state of the system via the environmental measurements.  In general, this
evolution is governed by the Schrödinger equation with a **non-Hermitian**
effective Hamiltonian

.. math::
    :label: heff

    H_{\rm eff}=H_{\rm sys}-\frac{i\hbar}{2}\sum_{i}C^{+}_{n}C_{n},

where again, the :math:`C_{n}` are collapse operators, each corresponding to a
separate irreversible process with rate :math:`\gamma_{n}`.  Here, the strictly
negative non-Hermitian portion of Eq. :eq:`heff` gives rise to a reduction in
the norm of the wave function, that to first-order in a small time
:math:`\delta t`, is given by
:math:`\left<\psi(t+\delta t)|\psi(t+\delta t)\right>=1-\delta p` where

.. math::
    :label: jump

    \delta p =\delta t \sum_{n}\left<\psi(t)|C^{+}_{n}C_{n}|\psi(t)\right>,

and :math:`\delta t` is such that :math:`\delta p \ll 1`.  With a probability
of remaining in the state :math:`\left|\psi(t+\delta t)\right>` given by
:math:`1-\delta p`, the corresponding quantum jump probability is thus Eq.
:eq:`jump`.  If the environmental measurements register a quantum jump, say via
the emission of a photon into the environment, or a change in the spin of a
quantum dot, the wave function undergoes a jump into a state defined by
projecting :math:`\left|\psi(t)\right>` using the collapse operator
:math:`C_{n}` corresponding to the measurement

.. math::
    :label: project

    \left|\psi(t+\delta t)\right>=C_{n}\left|\psi(t)\right>/\left<\psi(t)|C_{n}^{+}C_{n}|\psi(t)\right>^{1/2}.

If more than a single collapse operator is present in Eq. :eq:`heff`, the
probability of collapse due to the :math:`i\mathrm{th}`-operator :math:`C_{i}`
is given by

.. math::
    :label: pcn

    P_{i}(t)=\left<\psi(t)|C_{i}^{+}C_{i}|\psi(t)\right>/\delta p.

Evaluating the MC evolution to first-order in time is quite tedious.  Instead,
QuTiP uses the following algorithm to simulate a single realization of a quantum system.
Starting from a pure state :math:`\left|\psi(0)\right>`:

- **Ia:** Choose a random number :math:`r_1` between zero and one, representing
  the probability that a quantum jump occurs.

- **Ib:** Choose a random number :math:`r_2` between zero and one, used to
  select which collapse operator was responsible for the jump.

- **II:** Integrate the Schrödinger equation, using the effective Hamiltonian
  :eq:`heff` until a time :math:`\tau` such that the norm of the wave function
  satisfies :math:`\left<\psi(\tau)\right.\left|\psi(\tau)\right> = r_1`, at
  which point a jump occurs.

- **III:** The resultant jump projects the system at time :math:`\tau` into one
  of the renormalized states given by Eq. :eq:`project`.  The corresponding
  collapse operator :math:`C_{n}` is chosen such that :math:`n` is the smallest
  integer satisfying:

.. math::
    :label: mc3

    \sum_{i=1}^{n} P_{n}(\tau) \ge r_2

where the individual :math:`P_{n}` are given by Eq. :eq:`pcn`.  Note that the
left hand side of Eq. :eq:`mc3` is, by definition, normalized to unity.

- **IV:** Using the renormalized state from step III as the new initial
  condition at time :math:`\tau`, draw a new random number, and repeat the
  above procedure until the final simulation time is reached.


.. _monte-qutip:

Monte Carlo in QuTiP
====================

In QuTiP, Monte Carlo evolution is implemented with the :func:`.mcsolve`
function. It takes nearly the same arguments as the :func:`.mesolve`
function for master-equation evolution, except that the initial state must be a
ket vector, as oppose to a density matrix, and there is an optional keyword
parameter ``ntraj`` that defines the number of stochastic trajectories to be
simulated.  By default, ``ntraj=500`` indicating that 500 Monte Carlo
trajectories will be performed.

To illustrate the use of the Monte Carlo evolution of quantum systems in QuTiP,
let's again consider the case of a two-level atom coupled to a leaky cavity.
The only differences to the master-equation treatment is that in this case we
invoke the :func:`.mcsolve` function instead of :func:`.mesolve`

.. plot::
    :context: reset

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

The advantage of the Monte Carlo method over the master equation approach is that
only the state vector is required to be kept in the computers memory, as opposed
to the entire density matrix. For large quantum system this becomes a significant
advantage, and the Monte Carlo solver is therefore generally recommended for such
systems. For example, simulating a Heisenberg spin-chain consisting of 10 spins
with random parameters and initial states takes almost 7 times longer using the
master equation rather than Monte Carlo approach with the default number of
trajectories running on a quad-CPU machine.  Furthermore, it takes about 7 times
the memory as well. However, for small systems, the added overhead of averaging
a large number of stochastic trajectories to obtain the open system dynamics, as
well as starting the multiprocessing functionality, outweighs the benefit of the
minor (in this case) memory saving. Master equation methods are therefore
generally more efficient when Hilbert space sizes are on the order of a couple
of hundred states or smaller.


Monte Carlo Solver Result
-------------------------

The Monte Carlo solver returns a :class:`.McResult` object consisting of
expectation values and/or states. The main difference with :func:`.mesolve`'s
:class:`.Result` is that it optionally stores the result of each trajectory
together with their averages with the use of the ``"keep_runs_results"`` option.
When trajectories are stored, ``result.runs_expect`` is a list over the
expectation operators, trajectories and times in that order.
The averages are stored in ``result.average_expect`` and the standard derivation
of the expectation values in ``result.std_expect``. When the states are returned,
``result.runs_states`` will be an array of length ``ntraj``. Each element
contains an array of "Qobj" type ket with the same number of elements as ``times``.
``result.average_states`` is a list of density matrices computed as the average
of the states at each time step. Furthermore, the output will also contain a
list of times at which the collapse occurred, and which collapse operators did
the collapse. These can be obtained in  ``result.col_times`` and
``result.col_which`` respectively.


.. _monte-ntraj:

Changing the Number of Trajectories
-----------------------------------

By default, the ``mcsolve`` function runs 500 trajectories.
This value was chosen because it gives good accuracy, Monte Carlo errors scale
as :math:`1/n` where :math:`n` is the number of trajectories, and simultaneously
does not take an excessive amount of time to run. However, you can change the
number of trajectories to fit your needs. In order to run 1000 trajectories in
the above example, we can simply modify the call to ``mcsolve`` like:

.. code-block::

    data = mcsolve(H, psi0, times, c_ops e_ops=e_ops, ntraj=1000)

where we have added the keyword argument ``ntraj=1000`` at the end of the inputs.
Now, the Monte Carlo solver will calculate expectation values for both operators,
``a.dag() * a, sm.dag() * sm`` averaging over 1000 trajectories.


Other than a target number of trajectories, it is possible to use a computation
time or errors bars as condition to stop computing trajectories.

``timeout`` is quite simple as ``mcsolve`` will stop starting the computation of
new trajectories when it is reached. Thus:


.. code-block::

    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], e_ops=e_ops, ntraj=1000, timeout=60)

Will compute 60 seconds of trajectories or 1000, which ever is reached first.
The solver will finish any trajectory started when the timeout is reached. Therefore
if the computation time of a single trajectory is quite long, the overall computation
time can be much longer that the provided timeout.

Lastly, ``mcsolve`` can be instructed to stop when the statistical error of the
expectation values get under a certain value. When computing the average over
trajectories, the error on these are computed using
`jackknife resampling <https://en.wikipedia.org/wiki/Jackknife_resampling>`_
for each expect and each time and the computation will be stopped when all these values
are under the tolerance passed to ``target_tol``. Therefore:

.. code-block::

    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], e_ops=e_ops,
                   ntraj=1000, target_tol=0.01, timeout=600)

will stop either after all errors bars on expectation values are under ``0.01``, 1000
trajectories are computed or 10 minutes have passed, whichever comes first. When a
single values is passed, it is used as the absolute value of the tolerance.
When a pair of values is passed, it is understood as an absolute and relative
tolerance pair. For even finer control, one such pair can be passed for each ``e_ops``.
For example:

.. code-block::

    data = mcsolve(H, psi0, times, c_ops, e_ops=e_ops,  target_tol=[
        (1e-5, 0.1),
        (0, 0),
    ])

will stop when the error bars on the expectation values of the first ``e_ops`` are
under 10% of their average values.

If after computation of some trajectories, it is determined that more are needed, it
is possible to add trajectories to existing result by adding result together:

.. code-block::

    >>> run1 = mcsolve(H, psi, times, c_ops, e_ops=e_ops, ntraj=25)
    >>> print(run1.num_trajectories)
    25
    >>> run2 = mcsolve(H, psi, times, c_ops, e_ops=e_ops, ntraj=25)
    >>> print(run2.num_trajectories)
    25
    >>> merged = run1 + run2
    >>> print(merged.num_trajectories)
    50

Note that this merging operation only checks that the result are compatible --
i.e. that the ``e_ops`` and ``tlist`` are the same. It does not check that the same initial state or
Hamiltonian where used.


This can be used to explore the convergence of the Monte Carlo solver.
For example, the following code block plots expectation values for 1, 10 and 100
trajectories:

.. plot::
    :context: close-figs

    solver = MCSolver(H, c_ops=[np.sqrt(0.1) * a])
    c_ops=[np.sqrt(0.1) * a]
    e_ops = [a.dag() * a, sm.dag() * sm]

    data1 = mcsolve(H, psi0, times, c_ops, e_ops=e_ops, ntraj=1)
    data10 = data1 + mcsolve(H, psi0, times, c_ops, e_ops=e_ops, ntraj=9)
    data100 = data10 + mcsolve(H, psi0, times, c_ops, e_ops=e_ops, ntraj=90)

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


Mixed Initial states
--------------------

The Monte-Carlo solver can be used for mixed initial states. For example, if a
qubit can initially be in the excited state :math:`|+\rangle` with probability
:math:`p` or in the ground state :math:`|-\rangle` with probability
:math:`(1-p)`, the initial state is described by the density matrix
:math:`\rho_0 = p | + \rangle\langle + | + (1-p) | - \rangle\langle - |`.

In QuTiP, this initial density matrix can be created as follows:

.. code-block::

    ground = qutip.basis(2, 0)
    excited = qutip.basis(2, 1)
    density_matrix = p * excited.proj() + (1 - p) * ground.proj()

One can then pass this density matrix directly to ``mcsolve``, as in

.. code-block::

    mcsolve(H, density_matrix, ...)

Alternatively, using the class interface, if ``solver`` is an
:class:`.MCSolver` object, one can either call
``solver.run(density_matrix, ...)`` or pass the list of initial states like

.. code-block::

    solver.run([(excited, p), (ground, 1-p)], ...)

The number of trajectories can still be specified as a single number ``ntraj``.
In that case, QuTiP will automatically decide how many trajectories to use for
each of the initial states, guaranteeing that the total number of trajectories
is exactly the specified number. When using the class interface and providing
the initial state as a list, the `ntraj` parameter may also be a list
specifying the number of trajectories to use for each state manually. In either
case, the resulting :class:`McResult` will have attributes ``initial_states``
and ``ntraj_per_initial_state`` listing the initial states and the
corresponding numbers of trajectories that were actually used.

Note that in general, the fraction of trajectories starting in a given initial
state will (and can) not exactly match the probability :math:`p` of that state
in the initial ensemble. In this case, QuTiP will automatically apply a
correction to the averages, weighting for example the initial states with
"too few" trajectories more strongly. Therefore, the initial state returned in
the result object will always match the provided one up to numerical
inaccuracies. Furthermore, the result returned by the `mcsolve` call above is
equivalent to the following:

.. code-block::

    result1 = qutip.mcsolve(H, excited, ...)
    result2 = qutip.mcsolve(H, ground, ...)
    result1.merge(result2, p)

However, the single ``mcsolve`` call allows for more parallelization (see
below).

The Monte-Carlo solver with a mixed initial state currently does not support
specifying a target tolerance. Also, in case the simulation ends early due to
timeout, it is not guaranteed that all initial states have been sampled. If
not all initial states have been sampled, the resulting states will not be
normalized, and the result should be discarded.

Finally note that what we just discussed concerns the case of mixed initial
states where the provided Hamiltonian is an operator. If it is a superoperator
(i.e., a Liouvillian), ``mcsolve`` will generate trajectories of mixed states
(see below) and the present discussion does not apply.


Using the Improved Sampling Algorithm
-------------------------------------

Oftentimes, quantum jumps are rare. This is especially true in the context of
simulating gates for quantum information purposes, where typical gate times are
orders of magnitude smaller than typical timescales for decoherence. In this case,
using the standard monte-carlo sampling algorithm, we often repeatedly sample the
no-jump trajectory. We can thus reduce the number of required runs by only
sampling the no-jump trajectory once. We then extract the no-jump probability
:math:`p`, and for all future runs we only sample random numbers :math:`r_1`
where :math:`r_1>p`, thus ensuring that a jump will occur. When it comes time to
compute expectation values, we weight the no-jump trajectory by :math:`p` and
the jump trajectories by :math:`1-p`. This algorithm is described in [Abd19]_
and can be utilized by setting the option ``"improved_sampling"`` in the call
to ``mcsolve``:

.. plot::
    :context: close-figs

    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], options={"improved_sampling": True})

where in this case the first run samples the no-jump trajectory, and the
remaining 499 trajectories are all guaranteed to include (at least) one jump.

The power of this algorithm is most obvious when considering systems that rarely
undergo jumps. For instance, consider the following T1 simulation of a qubit with
a lifetime of 10 microseconds (assuming time is in units of nanoseconds)


.. plot::
    :context: close-figs

    times = np.linspace(0.0, 300.0, 100)
    psi0 = fock(2, 1)
    sm = fock(2, 0) * fock(2, 1).dag()
    omega = 2.0 * np.pi * 1.0
    H0 = -0.5 * omega * sigmaz()
    gamma = 1/10000
    data = mcsolve(
        [H0], psi0, times, [np.sqrt(gamma) * sm], e_ops=[sm.dag() * sm], ntraj=100
    )
    data_imp = mcsolve(
        [H0], psi0, times, [np.sqrt(gamma) * sm], e_ops=[sm.dag() * sm], ntraj=100,
        options={"improved_sampling": True}
    )

    plt.figure()
    plt.plot(times, data.expect[0], label="original")
    plt.plot(times, data_imp.expect[0], label="improved sampling")
    plt.plot(times, np.exp(-gamma * times), label=r"$\exp(-\gamma t)$")
    plt.title('Monte Carlo: improved sampling algorithm')
    plt.xlabel("time [ns]")
    plt.ylabel(r"$p_{1}$")
    plt.legend()
    plt.show()


The original sampling algorithm samples the no-jump trajectory on average 96.7%
of the time, while the improved sampling algorithm only does so once.


.. _monte-seeds:

Reproducibility
---------------

For reproducibility of Monte-Carlo computations it is possible to set the seed of the random
number generator:

.. code-block::

    >>> res1 = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, seeds=1, ntraj=1)
    >>> res2 = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, seeds=1, ntraj=1)
    >>> res3 = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, seeds=2, ntraj=1)
    >>> np.allclose(res1, res2)
    True
    >>> np.allclose(res1, res3)
    False

The ``seeds`` parameter can either be an integer or a numpy ``SeedSequence``, which
will then be used to create seeds for each trajectory. Alternatively it may be a list of
intergers or ``SeedSequence`` s with one seed for each trajectories. Seeds available in
the result object can be used to redo the same evolution:


.. code-block::

    >>> res1 = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=10)
    >>> res2 = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, seeds=res1.seeds, ntraj=10)
    >>> np.allclose(res1, res2)
    True


.. _monte-parallel:

Running trajectories in parallel
--------------------------------

Monte-Carlo evolutions often need hundreds of trajectories to obtain sufficient
statistics. Since all trajectories are independent of each other, they can be computed
in parallel. The option ``map`` can take ``"serial"``, ``"parallel"``, ``"loky"`` or ``"mpi"``.
Both ``"parallel"`` and ``"loky"`` compute trajectories on multiple CPUs using
respectively the `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
and `loky <https://loky.readthedocs.io/en/stable/index.html>`_ python modules.
The ``"mpi"`` option is for computing trajectories in a computing cluster, see the :ref:`MPI section<monte-mpi>` below.

.. code-block::

    >>> res_par = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, options={"map": "parallel"}, seeds=1)
    >>> res_ser = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, options={"map": "serial"}, seeds=1)
    >>> np.allclose(res_par.average_expect, res_ser.average_expect)
    True

Note that when running in parallel, the order in which the trajectories are added
to the result can differ. Therefore

.. code-block::

    >>> print(res_par.seeds[:3])
    [SeedSequence(entropy=1,spawn_key=(1,),),
     SeedSequence(entropy=1,spawn_key=(0,),),
     SeedSequence(entropy=1,spawn_key=(2,),)]

    >>> print(res_ser.seeds[:3])
    [SeedSequence(entropy=1,spawn_key=(0,),),
     SeedSequence(entropy=1,spawn_key=(1,),),
     SeedSequence(entropy=1,spawn_key=(2,),)]


Photocurrent
------------

The photocurrent, previously computed using the ``photocurrent_sesolve`` and
``photocurrent_sesolve`` functions, are now included in the output of
:func:`.mcsolve` as ``result.photocurrent``.


.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 8))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))
    e_ops = [a.dag() * a, sm.dag() * sm]
    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], e_ops=e_ops)

    plt.figure()
    plt.plot((times[:-1] + times[1:])/2, data.photocurrent[0])
    plt.title('Monte Carlo Photocurrent')
    plt.xlabel('Time')
    plt.ylabel('Photon detections')
    plt.show()


.. openmcsolve:

Open Systems
------------

``mcsolve`` can be used to study systems which have measurement and dissipative
interactions with their environment.  This is done by passing a Liouvillian including the
dissipative interaction to the solver instead of a Hamiltonian.
In this case the effective Liouvillian becomes:

.. math::
    :label: Leff

    L_{\rm eff}\rho = L_{\rm sys}\rho -\frac{1}{2}\sum_{i}\left( C^{+}_{n}C_{n}\rho + \rho C^{+}_{n}C_{n}\right),

With the collapse probability becoming:

.. math::
    :label: L_jump

    \delta p =\delta t \sum_{n}\mathrm{tr}\left(\rho(t)C^{+}_{n}C_{n}\right),

And a jump with the collapse operator ``n`` changing the state as:

.. math::
    :label: L_project

    \rho(t+\delta t) = C_{n} \rho(t) C^{+}_{n} / \mathrm{tr}\left( C_{n} \rho(t) C^{+}_{n} \right),


We can redo the previous example for a situation where only half the emitted photons are detected.

.. plot::
    :context: close-figs

    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 8))
    a  = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))
    H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
    L = liouvillian(H, [np.sqrt(0.05) * a])
    data = mcsolve(L, psi0, times, [np.sqrt(0.05) * a], e_ops=[a.dag() * a, sm.dag() * sm])

    plt.figure()
    plt.plot((times[:-1] + times[1:])/2, data.photocurrent[0])
    plt.title('Monte Carlo Photocurrent')
    plt.xlabel('Time')
    plt.ylabel('Photon detections')
    plt.show()


.. _monte-mpi:

Distributed Simulations Using MPI
=================================

..
    adapted from the `nm_mcsolve` tutorial notebook

Sometimes, many trajectories are needed to see the convergence of the trajectory average.
Using QuTiP's MPI capabilities, large numbers of trajectories can be computed in parallel
on multiple nodes of a computing cluster. On the QuTiP side, running Monte Carlo simulations
through MPI is as easy as replacing ``"map": "parallel"`` by ``"map": "mpi"`` in the provided options.
In addition, one should always provide the ``"num_cpus"`` option, which in this case specifies
the number of available worker processes. The number of available worker processes is typically one
less than the total number of processes assigned to the task.

The call to the Monte Carlo solver might look like this (for a more detailed example, see e.g.
`this tutorial notebook <https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v5/time-evolution/013_nonmarkovian_monte_carlo.ipynb>`_):

.. code-block:: python

    qutip.mcsolve(H, psi0, times, c_ops, ntraj=NTRAJ,
                  options={'store_states': True,
                           'progress_bar': False,
                           'map': 'mpi',
                           'num_cpus': NUM_WORKER_PROCESSES})

To invoke the MPI API, QuTiP relies on the ``MPIPoolExecutor`` class from the `mpi4py <https://mpi4py.github.io/>`_ module.
For instructions on how to set up an environment in which an ``MPIPoolExecutor`` can successfully
be created and communicate across nodes, we generally refer to the
`documentation of mpi4py <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html>`_ and to your system administrator.

Below, we provide an example batch script that can be submitted to a SLURM workload manager. The authors of this guide
used this script to perform a parallel calculation on 500 CPUs distributed over 5 nodes of the supercomputer
`HOKUSAI <https://www.r-ccs.riken.jp/exhibit_contents/SC20/hokusai.html>`_, using the `MPICH <https://www.mpich.org>`_
implementation of the MPI standard. However, one should expect that adjustments to the script are required depending
on the available MPI implementations and their versions, as well as the workload manager and its version and configuration.

.. code-block:: bash

    #!/bin/bash
    #SBATCH --partition=XXXXX
    #SBATCH --account=XXXXX

    #SBATCH --nodes=5
    #SBATCH --ntasks=501
    #SBATCH --mem-per-cpu=1G

    #SBATCH --time=0-10:00

    source ~/.bashrc

    module purge
    module load mpi/mpich-x86_64
    conda activate qutip-environment

    mpirun -np $SLURM_NTASKS -bind-to core python -m mpi4py.futures XXXXX.py



.. plot::
    :context: reset
    :include-source: false
    :nofigs:

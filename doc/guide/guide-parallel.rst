.. _guide-parallel:

***************
Parallelization
***************

QuTiP provides several ways to run independent calculations concurrently.
Choose an interface based on the work you want to distribute:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Work
     - Interface
     - Examples and documentation
   * - Trajectories
     - The solver's ``map`` option
     - :ref:`Monte Carlo guide <monte-parallel>`; :ref:`MPI example <mpi-trajectories>`
   * - Two-time correlations
     - ``map`` and ``map_kw`` arguments
     - :ref:`Correlation guide <correlation>`; :ref:`MPI example <mpi-correlations>`
   * - Independent calculations
     - :func:`.parallel_map`, :func:`.loky_pmap`, :func:`.mpi_pmap`, or :func:`.serial_map`
     - :ref:`MPI parameter sweep <mpi-independent>`
   * - Laguerre-method Wigner calculation
     - :func:`.wigner` with ``method="laguerre"`` and ``parfor=True``
     - :func:`.wigner` API documentation

Choosing a backend
==================

The available backends depend on the interface. ``serial`` runs tasks in the
current process; ``parallel`` uses Python's :ref:`multiprocessing
<multiprocessing-context>`; ``loky`` uses the optional `loky
<https://loky.readthedocs.io/en/stable/index.html>`_ package; and ``mpi`` uses
:ref:`MPI <monte-mpi>` for distributed calculations. For direct mapping,
choose the corresponding mapping function instead. See each function's
documentation for its supported options.
Starting workers and transferring data have overhead, so parallel execution
is not necessarily faster for small calculations.

If a multiprocessing calculation fails in a notebook or interactive session, see
:ref:`parallel-interactive-troubleshooting`.

.. _multiprocessing-context:

Multiprocessing contexts
========================

The ``"parallel"`` backend and :func:`.parallel_map` use Python's
``multiprocessing`` module to start worker processes. The *start method*
determines how those processes are created and what state they inherit.
QuTiP selects a multiprocessing context as follows:

* On **Python versions before 3.15**, QuTiP uses ``fork`` on Linux and macOS to
  preserve its previous behaviour. On other platforms it uses the application
  default.
* On **Python 3.15 and later**, QuTiP uses the application's multiprocessing
  default on all platforms. The application can select this default; otherwise
  Python chooses it according to the platform and Python version.

.. _parallel-context-override:

Overriding the multiprocessing context
--------------------------------------

You can explicitly select a multiprocessing context before starting a
calculation. For example, on a platform that supports it, the following
requests the ``fork`` method:

.. code-block:: python

    import multiprocessing as mp
    import qutip

    print(mp.get_all_start_methods())  # Methods available on this platform
    qutip.settings.mp_context = "fork"

    # Run the calculation using the "parallel" backend here.

    qutip.settings.mp_context = None  # Follow the application default again

The setting accepts an available start-method name, a context returned by
``multiprocessing.get_context()``, or ``None``. It affects QuTiP's
multiprocessing-backed parallel operations, not the ``loky`` or MPI backends.
Setting it to ``None`` selects the application default.

.. warning::
On POSIX systems, ``fork`` is unsafe if the parent process has multiple threads when workers start: a child can inherit locks or library state left by threads that no longer exist in the child. Threads may be started by notebook kernels, numerical libraries, or other dependencies, even if your code does not start them explicitly. Use ``fork`` only when the parent is single-threaded at that point and the libraries and resources used by workers are known to be safe after a fork.
On macOS, system libraries may start threads, so ``fork`` can cause subprocess crashes
even when the application does not explicitly create threads.


.. _monte-mpi:

Distributed calculations using MPI
===================================

..
    adapted from the `nm_mcsolve` tutorial notebook

MPI distributes independent tasks across worker processes, potentially on
multiple nodes. Trajectory solvers distribute trajectories, two-time
correlation functions distribute work over ``tlist``, and :func:`.mpi_pmap`
distributes calls to a user-defined function. A single :func:`.mesolve`
integration, for example, is not itself distributed across MPI processes.

Install ``mpi4py`` and a compatible MPI implementation in the environment
used by all workers. Specify the number of workers explicitly with
``num_cpus``.

.. _mpi-trajectories:

Trajectory solvers
------------------

For trajectory solvers, select MPI in the solver options:

.. code-block:: python

    qutip.mcsolve(H, psi0, times, c_ops, ntraj=NTRAJ,
                  options={'store_states': True,
                           'progress_bar': False,
                           'map': 'mpi',
                           'num_cpus': NUM_WORKER_PROCESSES})

For more on trajectory simulations and parallel execution, see the
:ref:`Monte Carlo guide <monte-parallel>`.

The same options work with :func:`.nm_mcsolve`, :func:`.smesolve`, and
:func:`.ssesolve`.

.. _mpi-correlations:

Two-time correlations
---------------------

For two-time correlations, given a Hamiltonian ``H``,
initial state ``rho0``, collapse operators ``c_ops``, and operator ``a``,
pass the map arguments directly:

.. code-block:: python

    corr = qutip.correlation_2op_2t(
        H, rho0, tlist, taulist, c_ops, a.dag(), a,
        map="mpi", map_kw={"num_cpus": 4},
    )

.. _mpi-independent:

Independent calculations
------------------------

Use :func:`.mpi_pmap` to distribute independent calculations. The following
parameter sweep calculates the final population for several damping rates:

.. code-block:: python

    import numpy as np
    import qutip as qt

    def simulate(gamma):
        a = qt.destroy(2)
        rho0 = qt.basis(2, 1).proj()
        result = qt.mesolve(
            0.5 * qt.sigmaz(), rho0, np.linspace(0, 5, 101),
            c_ops=[np.sqrt(gamma) * a], e_ops=[a.dag() * a],
        )
        return result.expect[0][-1]

    if __name__ == "__main__":
        rates = [0.1, 0.2, 0.5, 1.0]
        populations = qt.mpi_pmap(simulate, rates, map_kw={"num_cpus": 2})
        print(list(zip(rates, populations)))

Run this as a Python script if your MPI installation supports dynamic worker
spawning. The main guard prevents workers from starting another pool. Tasks and
their results must be serializable. See the mpi4py documentation linked below
for launcher examples.

MPI configuration and cluster execution
---------------------------------------

For trajectory solvers, ``mpi_options`` can pass additional keyword arguments
to ``MPIPoolExecutor``. For correlations and :func:`.mpi_pmap`, use
``map_kw``. If a worker count is not supplied to :func:`.mpi_pmap`, QuTiP
uses ``QUTIP_NUM_PROCESSES`` when set to a nonzero value; otherwise, it
detects a default process count. Set the environment variable before starting
Python.

To invoke the MPI API, QuTiP relies on the ``MPIPoolExecutor`` class from the `mpi4py <https://mpi4py.github.io/>`_ module.
For instructions on how to set up an environment in which an ``MPIPoolExecutor`` can successfully
be created and communicate across nodes, see the
`documentation of mpi4py <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html>`_.

Direct script execution normally starts workers dynamically. The cluster
example below instead launches ``mpi4py.futures`` with a fixed set of MPI
processes: one is the master and the remaining processes are workers. The MPI
launcher creates the worker pool; set ``NUM_WORKER_PROCESSES`` to the number
of launched processes minus one so QuTiP's setting matches the available
workers. Launch requirements and support for dynamic spawning depend on the
MPI installation and scheduler.

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



Troubleshooting
===============

.. _parallel-interactive-troubleshooting:

Parallel execution in notebooks and interactive sessions
--------------------------------------------------------

Some multiprocessing calculations in notebooks or other interactive sessions
may depend on state or functions defined in the session. With the
multiprocessing ``spawn`` or ``forkserver`` start method, workers must be able
to reconstruct the work sent to them;
functions defined interactively are not always importable by the workers.
Consequently, a calculation that worked with ``fork`` may fail when the
default context changes. This depends on the calculation and its inputs, not
just on whether it runs in a notebook. See Python's documentation on
`contexts and start methods <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.

If a multiprocessing calculation fails in an interactive session, consider the
following options:

* Try the ``"loky"`` backend, which supports many interactively defined
  functions. Install the optional ``loky`` package first, then pass
  ``options={"map": "loky"}`` to a trajectory solver, or use
  :func:`.loky_pmap` in place of :func:`.parallel_map`. The objects used by
  the calculation must still be serializable.
* Move worker functions and any required state into an importable Python
  module. When running from a script with ``spawn`` or ``forkserver``, protect
  the code that starts parallel work with ``if __name__ == "__main__":``.
* Use ``"serial"`` if the calculation does not need to run in parallel.



.. _guide-parallel:

***************
Parallelization
***************

QuTiP provides several ways to run independent calculations concurrently. The
interface depends on the function being called:

* Trajectory solvers :func:`.mcsolve`, :func:`.nm_mcsolve`,
  :func:`.smesolve`, and :func:`.ssesolve` accept a ``map`` solver option.
  For example, use ``options={"map": "parallel"}`` or
  ``options={"map": "loky"}``. The corresponding solver classes also accept
  this option. See :ref:`monte-parallel` for a trajectory example.
* :func:`.correlation_2op_2t`, :func:`.correlation_3op_2t`, and
  :func:`.correlation_3op` accept ``map`` and ``map_kw`` directly, for
  example ``map="parallel"`` and ``map_kw={"num_cpus": 2}``.
* For a general mapping operation, use :func:`.parallel_map`,
  :func:`.loky_pmap`, :func:`.mpi_pmap`, or :func:`.serial_map` directly.
* Some functions expose a separate switch for parallel work. For example,
  :func:`.wigner` uses ``parfor=True`` with ``method="laguerre"`` to use
  :func:`.parallel_map`.

``parallel`` uses Python's multiprocessing module, ``loky`` uses the optional
`loky <https://loky.readthedocs.io/en/stable/index.html>`_ package, and ``mpi``
uses MPI for distributed computation. The ``serial`` option runs without a
process pool. Availability and supported options vary by caller; consult the
function's documentation. For MPI setup and trajectory execution, see
:ref:`monte-mpi`. Starting workers and transferring data have overhead, so
parallel execution is not necessarily faster for small calculations.

If a parallel calculation fails in a notebook or interactive session, see
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

``fork`` is unavailable on Windows. It can cause hangs or crashes when the
parent process has multiple threads, and Python considers it unsafe on macOS.
Notebook kernels and numerical libraries may start threads, so use this
override only after considering those risks. It also does not remove the need
to serialize tasks submitted to :func:`.parallel_map`.

The setting accepts an available start-method name, a context returned by
``multiprocessing.get_context()``, or ``None``. It affects QuTiP's
multiprocessing-backed parallel operations, not the ``loky`` or MPI backends.
Setting it to ``None`` selects the application default even on older Python
versions; it does not restore QuTiP's legacy ``fork`` selection. Set it before
starting the parallel calculation.

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



Troubleshooting
===============

.. _parallel-interactive-troubleshooting:

Parallel execution in notebooks and interactive sessions
--------------------------------------------------------

Some parallel calculations in notebooks or other interactive sessions may
depend on state or functions defined in the session. With ``spawn`` or
``forkserver``, workers must be able to reconstruct the work sent to them;
functions defined interactively are not always importable by the workers.
Consequently, a calculation that worked with ``fork`` may fail when the
default context changes. This depends on the calculation and its inputs, not
just on whether it runs in a notebook. See Python's documentation on
`contexts and start methods <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.

If a parallel calculation fails in an interactive session, consider the
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



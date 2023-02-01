.. _parfor:

******************************************
Parallel computation
******************************************

Parallel map and parallel for-loop
----------------------------------

Often one is interested in the output of a given function as a single-parameter is varied.
For instance, we can calculate the steady-state response of our system as the driving frequency is varied.
In cases such as this, where each iteration is independent of the others, we can speedup the calculation by performing the iterations in parallel.
In QuTiP, parallel computations may be performed using the :func:`qutip.solver.parallel.parallel_map` function.

To use the this function we need to define a function of one or more variables, and the range over which one of these variables are to be evaluated. For example:


.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

   >>> result = parallel_map(func1, range(10))

   >>> result_array = np.array(result)

   >>> print(result_array[:, 0])  # == a
   [0 1 2 3 4 5 6 7 8 9]

   >>> print(result_array[:, 1])  # == b
   [ 0  1  4  9 16 25 36 49 64 81]

   >>> print(result_array[:, 2])  # == c
   [  0   1   8  27  64 125 216 343 512 729]

   >>> print(result)
   [(0, 0, 0), (1, 1, 1), (2, 4, 8), (3, 9, 27), (4, 16, 64)]


The :func:`qutip.solver.parallel.parallel_map` function is not limited to just numbers, but also works for a variety of outputs:

.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

   >>> def func2(x): return x, Qobj(x), 'a' * x

   >>> results = parallel_map(func2, range(5))

   >>> print([result[0] for result in results])
   [0 1 2 3 4]

   >>> print([result[1] for result in results])
   [Quantum object: dims = [[1], [1]], shape = (1, 1), type = bra
   Qobj data =
   [[0.]]
    Quantum object: dims = [[1], [1]], shape = (1, 1), type = bra
   Qobj data =
   [[1.]]
    Quantum object: dims = [[1], [1]], shape = (1, 1), type = bra
   Qobj data =
   [[2.]]
    Quantum object: dims = [[1], [1]], shape = (1, 1), type = bra
   Qobj data =
   [[3.]]
    Quantum object: dims = [[1], [1]], shape = (1, 1), type = bra
   Qobj data =
   [[4.]]]

   >>>print([result[2] for result in results])
   ['' 'a' 'aa' 'aaa' 'aaaa']


One can also define functions with **multiple** input arguments and keyword arguments.

.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

    >>> def sum_diff(x, y, z=0): return x + y, x - y, z

    >>> parallel_map(sum_diff, [1, 2, 3], task_args=(np.array([4, 5, 6]),), task_kwargs=dict(z=5.0))
    [(array([5, 6, 7]), array([-3, -4, -5]), 5.0),
     (array([6, 7, 8]), array([-2, -3, -4]), 5.0),
     (array([7, 8, 9]), array([-1, -2, -3]), 5.0)]


The :func:`qutip.solver.parallel.parallel_map` function supports progressbar by setting the keyword argument `progress_bar` to `True`.
The number of cpu used can also be controlled using the `map_kw` keyword, per default, all available cpus are used.

.. doctest::
  :options: +SKIP

   >>> import time

   >>> def func(x): time.sleep(1)

   >>> result = parallel_map(func, range(50), progress_bar=True, map_kw={"num_cpus": 2})

   10.0%. Run time:   3.10s. Est. time left: 00:00:00:27
   20.0%. Run time:   5.11s. Est. time left: 00:00:00:20
   30.0%. Run time:   8.11s. Est. time left: 00:00:00:18
   40.0%. Run time:  10.15s. Est. time left: 00:00:00:15
   50.0%. Run time:  13.15s. Est. time left: 00:00:00:13
   60.0%. Run time:  15.15s. Est. time left: 00:00:00:10
   70.0%. Run time:  18.15s. Est. time left: 00:00:00:07
   80.0%. Run time:  20.15s. Est. time left: 00:00:00:05
   90.0%. Run time:  23.15s. Est. time left: 00:00:00:02
   100.0%. Run time:  25.15s. Est. time left: 00:00:00:00
   Total run time:  28.91s

There is a function called :func:`qutip.solver.parallel.serial_map` that works as a non-parallel drop-in replacement for :func:`qutip.solver.parallel.parallel_map`, which allows easy switching between serial and parallel computation.
Qutip also has the function :func:`qutip.solver.parallel.loky_map` as another drop-in replacement. It use the `loky` module instead of `multiprocessing` to run in parallel.
Parallel processing is useful for repeated tasks such as generating plots corresponding to the dynamical evolution of your system, or simultaneously simulating different parameter configurations.


IPython-based parallel_map
--------------------------

When QuTiP is used with IPython interpreter, there is an alternative parallel for-loop implementation in the QuTiP  module :func:`qutip.ipynbtools`, see :func:`qutip.ipynbtools.parallel_map`. The advantage of this parallel_map implementation is based on IPython's powerful framework for parallelization, so the compute processes are not confined to run on the same host as the main process.

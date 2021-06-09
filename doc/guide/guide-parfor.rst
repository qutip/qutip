.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _parfor:

******************************************
Parallel computation
******************************************

Parallel map and parallel for-loop
----------------------------------

Often one is interested in the output of a given function as a single-parameter is varied. For instance, we can calculate the steady-state response of our system as the driving frequency is varied.  In cases such as this, where each iteration is independent of the others, we can speedup the calculation by performing the iterations in parallel. In QuTiP, parallel computations may be performed using the :func:`qutip.parallel.parallel_map` function or the :func:`qutip.parallel.parfor` (parallel-for-loop) function.

To use the these functions we need to define a function of one or more variables, and the range over which one of these variables are to be evaluated. For example:


.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

   >>> def func1(x): return x, x**2, x**3

   >>> a, b, c = parfor(func1, range(10))

   >>> print(a)
   [0 1 2 3 4 5 6 7 8 9]

   >>> print(b)
   [ 0  1  4  9 16 25 36 49 64 81]

   >>> print(c)
   [  0   1   8  27  64 125 216 343 512 729]

or

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


Note that the return values are arranged differently for the :func:`qutip.parallel.parallel_map` and the :func:`qutip.parallel.parfor` functions, as illustrated below. In particular, the return value of :func:`qutip.parallel.parallel_map` is not enforced to be NumPy arrays, which can avoid unnecessary copying if all that is needed is to iterate over the resulting list:


.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

   >>> result = parfor(func1, range(5))

   >>> print(result)
   [array([0, 1, 2, 3, 4]), array([ 0,  1,  4,  9, 16]), array([ 0,  1,  8, 27, 64])]

   >>> result = parallel_map(func1, range(5))

   >>> print(result)
   [(0, 0, 0), (1, 1, 1), (2, 4, 8), (3, 9, 27), (4, 16, 64)]

The :func:`qutip.parallel.parallel_map` and :func:`qutip.parallel.parfor` functions are not limited to just numbers, but also works for a variety of outputs:

.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

   >>> def func2(x): return x, Qobj(x), 'a' * x

   >>> a, b, c = parfor(func2, range(5))

   >>> print(a)
   [0 1 2 3 4]

   >>> print(b)
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

   >>>print(c)
   ['' 'a' 'aa' 'aaa' 'aaaa']


One can also define functions with **multiple** input arguments and even keyword arguments. Here the :func:`qutip.parallel.parallel_map` and :func:`qutip.parallel.parfor` functions behaves differently:
While :func:`qutip.parallel.parallel_map` only iterate over the values `arguments`, the :func:`qutip.parallel.parfor` function simultaneously iterates over all arguments:

.. doctest::
  :skipif: not os_nt
  :options: +NORMALIZE_WHITESPACE

    >>> def sum_diff(x, y, z=0): return x + y, x - y, z

    >>> parfor(sum_diff, [1, 2, 3], [4, 5, 6], z=5.0)
    [array([5, 7, 9]), array([-3, -3, -3]), array([5., 5., 5.])]

    >>> parallel_map(sum_diff, [1, 2, 3], task_args=(np.array([4, 5, 6]),), task_kwargs=dict(z=5.0))
    [(array([5, 6, 7]), array([-3, -4, -5]), 5.0),
     (array([6, 7, 8]), array([-2, -3, -4]), 5.0),
     (array([7, 8, 9]), array([-1, -2, -3]), 5.0)]

Note that the keyword arguments can be anything you like, but the keyword values are **not** iterated over. The keyword argument *num_cpus* is reserved as it sets the number of CPU's used by parfor. By default, this value is set to the total number of physical processors on your system. You can change this number to a lower value, however setting it higher than the number of CPU's will cause a drop in performance. In :func:`qutip.parallel.parallel_map`, keyword arguments to the task function are specified using `task_kwargs` argument, so there is no special reserved keyword arguments.

The :func:`qutip.parallel.parallel_map` function also supports progressbar, using the keyword argument `progress_bar` which can be set to `True` or to an instance of :class:`qutip.ui.progressbar.BaseProgressBar`. There is a function called :func:`qutip.parallel.serial_map` that works as a non-parallel drop-in replacement for :func:`qutip.parallel.parallel_map`, which allows easy switching between serial and parallel computation.

.. doctest::
  :options: +SKIP

   >>> import time

   >>> def func(x): time.sleep(1)

   >>> result = parallel_map(func, range(50), progress_bar=True)

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

Parallel processing is useful for repeated tasks such as generating plots corresponding to the dynamical evolution of your system, or simultaneously simulating different parameter configurations.


IPython-based parallel_map
--------------------------

When QuTiP is used with IPython interpreter, there is an alternative parallel for-loop implementation in the QuTiP  module :func:`qutip.ipynbtools`, see :func:`qutip.ipynbtools.parallel_map`. The advantage of this parallel_map implementation is based on IPython's powerful framework for parallelization, so the compute processes are not confined to run on the same host as the main process.

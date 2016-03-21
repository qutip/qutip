.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _parfor:

******************************************
Parallel computation
******************************************

Parallel map and parallel for-loop
----------------------------------

.. ipython::
   :suppress:

   In [1]: from qutip import *

Often one is interested in the output of a given function as a single-parameter is varied. For instance, we can calculate the steady-state response of our system as the driving frequency is varied.  In cases such as this, where each iteration is independent of the others, we can speedup the calculation by performing the iterations in parallel. In QuTiP, parallel computations may be performed using the :func:`qutip.parallel.parallel_map` function or the :func:`qutip.parallel.parfor` (parallel-for-loop) function.

To use the these functions we need to define a function of one or more variables, and the range over which one of these variables are to be evaluated. For example:


.. ipython-posix::

   In [1]: def func1(x): return x, x**2, x**3
   
   In [2]: a, b, c = parfor(func1, range(10))
   
   In [3]: print(a)
   
   In [4]: print(b)
   
   In [5]: print(c)
   
or

.. ipython-posix::

   In [1]: result = parallel_map(func1, range(10))
   
   In [2]: result_array = np.array(result)

   In [3]: print(result_array[:, 0])  # == a

   In [4]: print(result_array[:, 1])  # == b

   In [5]: print(result_array[:, 2])  # == c


Note that the return values are arranged differently for the :func:`qutip.parallel.parallel_map` and the :func:`qutip.parallel.parfor` functions, as illustrated below. In particular, the return value of :func:`qutip.parallel.parallel_map` is not enforced to be NumPy arrays, which can avoid unnecessary copying if all that is needed is to iterate over the resulting list:


.. ipython-posix::

   In [1]: result = parfor(func1, range(5))
   
   In [2]: print(result)

   In [3]: result = parallel_map(func1, range(5))
   
   In [4]: print(result)

The :func:`qutip.parallel.parallel_map` and :func:`qutip.parallel.parfor` functions are not limited to just numbers, but also works for a variety of outputs:

.. ipython-posix::

   In [1]: def func2(x): return x, Qobj(x), 'a' * x
   
   In [2]: a, b, c = parfor(func2, range(5))
   
   In [3]: print(a)
   
   In [4]: print(b)
   
   In [5]: print(c)


.. note::

    New in QuTiP 3.

One can also define functions with **multiple** input arguments and even keyword arguments. Here the :func:`qutip.parallel.parallel_map` and :func:`qutip.parallel.parfor` functions behaves differently:
While :func:`qutip.parallel.parallel_map` only iterate over the values `arguments`, the :func:`qutip.parallel.parfor` function simultaneously iterates over all arguments:

.. ipython-posix::
    
    In [1]: def sum_diff(x, y, z=0): return x + y, x - y, z
    
    In [2]: parfor(sum_diff, [1, 2, 3], [4, 5, 6], z=5.0)

    In [2]: parallel_map(sum_diff, [1, 2, 3], task_args=(np.array([4, 5, 6]),), task_kwargs=dict(z=5.0))

Note that the keyword arguments can be anything you like, but the keyword values are **not** iterated over. The keyword argument *num_cpus* is reserved as it sets the number of CPU's used by parfor. By default, this value is set to the total number of physical processors on your system. You can change this number to a lower value, however setting it higher than the number of CPU's will cause a drop in performance. In :func:`qutip.parallel.parallel_map`, keyword arguments to the task function are specified using `task_kwargs` argument, so there is no special reserved keyword arguments. 

The :func:`qutip.parallel.parallel_map` function also supports progressbar, using the keyword argument `progress_bar` which can be set to `True` or to an instance of :class:`qutip.ui.progressbar.BaseProgressBar`. There is a function called :func:`qutip.parallel.serial_map` that works as a non-parallel drop-in replacement for :func:`qutip.parallel.parallel_map`, which allows easy switching between serial and parallel computation.

.. ipython-posix::

   In [1]: import time

   In [2]: def func(x): time.sleep(1)

   In [2]: result = parallel_map(func, range(50), progress_bar=True)
   

Parallel processing is useful for repeated tasks such as generating plots corresponding to the dynamical evolution of your system, or simultaneously simulating different parameter configurations.


IPython-based parallel_map
--------------------------

.. note::

    New in QuTiP 3.

When QuTiP is used with IPython interpreter, there is an alternative parallel for-loop implementation in the QuTiP  module :func:`qutip.ipynbtools`, see :func:`qutip.ipynbtools.parallel_map`. The advantage of this parallel_map implementation is based on IPythons powerful framework for parallelization, so the compute processes are not confined to run on the same host as the main process. 


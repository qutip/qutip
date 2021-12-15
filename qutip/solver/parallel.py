"""
This module provides functions for parallel execution of loops and function
mappings, using the builtin Python module multiprocessing or the loky parallel
execution library.
"""
__all__ = ['parallel_map', 'serial_map', 'loky_pmap', 'get_map']

import multiprocessing
import os
import sys
import time
from qutip.ui.progressbar import progess_bars

if sys.platform == 'darwin':
    Pool = multiprocessing.get_context('fork').Pool
else:
    Pool = multiprocessing.Pool

default_map_kw = {
    'job_timeout': 1e32,  # Effectively infinite
    'timeout': 1e32,  # Effectively infinite
    'num_cpus': multiprocessing.cpu_count(),
}


def serial_map(task, values, task_args=None, task_kwargs=None, *,
               reduce_func=None, map_kw=None,
               progress_bar=None, progress_bar_kwargs={}):
    """
    Serial mapping function with the same call signature as parallel_map, for
    easy switching between serial and parallel execution. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    This function work as a drop-in replacement of :func:`qutip.parallel_map`.

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar
    map_kw:
        Other options

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for each
        value in ``values``.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)
    remaining_ntraj = len(values)
    map_kwargs = default_map_kw.copy()
    if isinstance(map_kw, dict):
        map_kwargs.update(map_kw)
    end_time = map_kwargs['timeout'] + time.time()
    results = []
    for n, value in enumerate(values):
        if time.time() > end_time:
            break
        result = task(value, *task_args, **task_kwargs)
        if reduce_func is not None:
            remaining_ntraj = reduce_func(result)
        else:
            results.append(result)
        if remaining_ntraj is not None and remaining_ntraj <= 0:
            end_time = 0
        progress_bar.update(n)
    progress_bar.finished()

    return results


def parallel_map(task, values, task_args=None, task_kwargs=None, *,
                 reduce_func=None, map_kw=None,
                 progress_bar=None, progress_bar_kwargs={}):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar
    map_kw:
        Other options

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    map_kwargs = default_map_kw.copy()
    if isinstance(map_kw, dict):
        map_kwargs.update(map_kw)
    end_time = map_kwargs['timeout'] + time.time()
    job_time = map_kwargs['job_timeout']

    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)
    remaining_ntraj = len(values)

    results = []
    try:
        pool = Pool(processes=map_kwargs['num_cpus'])

        async_res = [pool.apply_async(task, (value,) + task_args, task_kwargs)
                     for value in values]

        for job in async_res:
            remaining_time = min(end_time - time.time(), job_time)
            result = job.get(remaining_time)
            if reduce_func is not None:
                remaining_ntraj = reduce_func(result)
            else:
                results.append(result)
            if remaining_ntraj is not None and remaining_ntraj <= 0:
                job_time = 0
            progress_bar.update()

    except KeyboardInterrupt as e:
        raise e

    except multiprocessing.TimeoutError:
        pass

    finally:
        os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
        pool.terminate()
        pool.join()

    progress_bar.finished()
    return results


def loky_pmap(task, values, task_args=None, task_kwargs=None, *,
              reduce_func=None, map_kw=None,
              progress_bar=None, progress_bar_kwargs={}):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Use the loky module instead of multiprocessing.

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar
    **kwargs:
        Other options to pass to loky

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    from loky import get_reusable_executor, TimeoutError

    map_kwargs = default_map_kw.copy()
    if isinstance(map_kw, dict):
        map_kwargs.update(map_kw)

    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)

    executor = get_reusable_executor(max_workers=map_kwargs['num_cpus'])
    end_time = map_kwargs['timeout'] + time.time()
    job_time = map_kwargs['job_timeout']
    results = []
    remaining_ntraj = len(values)

    try:
        jobs = [executor.submit(task, value, *task_args, **task_kwargs)
               for value in values]

        for job in jobs:
            remaining_time = min(end_time - time.time(), job_time)
            result = job.result(remaining_time)
            if reduce_func is not None:
                remaining_ntraj = reduce_func(result)
            else:
                results.append(result)
            if remaining_ntraj is not None and remaining_ntraj <= 0:
                job_time = 0
            progress_bar.update()

    except KeyboardInterrupt as e:
        [job.cancel() for job in jobs]
        raise e

    except TimeoutError:
        [job.cancel() for job in jobs]

    finally:
        executor.shutdown()
    progress_bar.finished()
    os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
    return results


get_map = {
    True: parallel_map,
    'True': parallel_map,
    "parallel": parallel_map,
    "parallel_map": parallel_map,
    None: serial_map,
    False: serial_map,
    "False": serial_map,
    "serial": serial_map,
    "serial_map": serial_map,
    "loky": loky_pmap,
}

"""
This module provides functions for parallel execution of loops and function
mappings, using the builtin Python module multiprocessing or the loky parallel execution library.
"""
__all__ = ['parallel_map', 'serial_map', 'loky_pmap', 'get_map']

import multiprocessing
import os
import sys
import time
import threading
from qutip.ui.progressbar import progess_bars
from qutip.utilities import available_cpu_count

if sys.platform == 'darwin':
    Pool = multiprocessing.get_context('fork').Pool
else:
    Pool = multiprocessing.Pool


default_map_kw = {
    'job_timeout': threading.TIMEOUT_MAX,
    'timeout': threading.TIMEOUT_MAX,
    'num_cpus': available_cpu_count(),
}


def _read_map_kw(options):
    options = options or {}
    map_kw = default_map_kw.copy()
    map_kw.update({k: v for k, v in options.items() if v is not None})
    return map_kw


def serial_map(task, values, task_args=None, task_kwargs=None,
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
    reduce_func : func (optional)
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for 'timeout' the maximum time for the
        whole map.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for each
        value in ``values``. If a ``reduce_func`` is provided, and empty list
        will be returned.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    map_kw = _read_map_kw(map_kw)
    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)
    end_time = map_kw['timeout'] + time.time()
    results = []
    for n, value in enumerate(values):
        if time.time() > end_time:
            break
        progress_bar.update(n)
        result = task(value, *task_args, **task_kwargs)
        if reduce_func is not None:
            reduce_func(result)
        else:
            results.append(result)
    progress_bar.finished()

    return results


def parallel_map(task, values, task_args=None, task_kwargs=None,
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
    reduce_func : func (optional)
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for:
        'timeout': Maximum time for the whole map.
        'job_timeout': Maximum time for each job in the map.
        'num_cpus': Number of job to run at once.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    map_kw = _read_map_kw(map_kw)
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    end_time = map_kw['timeout'] + time.time()
    job_time = map_kw['job_timeout']

    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)

    results = []
    try:
        pool = Pool(processes=map_kw['num_cpus'])

        async_res = [pool.apply_async(task, (value,) + task_args, task_kwargs)
                     for value in values]

        for job in async_res:
            remaining_time = min(end_time - time.time(), job_time)
            result = job.get(remaining_time)
            if reduce_func is not None:
                reduce_func(result)
            else:
                results.append(result)
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


def loky_pmap(task, values, task_args=None, task_kwargs=None,
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
    reduce_func : func (optional)
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for:
        'timeout': Maximum time for the whole map.
        'job_timeout': Maximum time for each job in the map.
        'num_cpus': Number of job to run at once.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    map_kw = _read_map_kw(map_kw)
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    from loky import get_reusable_executor, TimeoutError

    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)

    executor = get_reusable_executor(max_workers=map_kw['num_cpus'])
    end_time = map_kw['timeout'] + time.time()
    job_time = map_kw['job_timeout']
    results = []

    try:
        jobs = [executor.submit(task, value, *task_args, **task_kwargs)
               for value in values]

        for job in jobs:
            remaining_time = min(end_time - time.time(), job_time)
            result = job.result(remaining_time)
            if reduce_func is not None:
                reduce_func(result)
            else:
                results.append(result)
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


def get_map(options):
    if "parallel" in options['map']:
        return parallel_map
    elif "serial" in options['map']:
        return serial_map
    elif "loky" in options['map']:
        return loky_pmap
    else:
        raise ValueError("map not found, available options are 'parallel',"
                         " 'serial' and 'loky'")

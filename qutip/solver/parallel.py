"""
This module provides functions for parallel execution of loops and function
mappings, using the builtin Python module multiprocessing or the loky parallel
execution library.
"""
__all__ = ['parallel_map', 'serial_map', 'loky_pmap']

import multiprocessing
import os
import sys
import time
import threading
import concurrent.futures
from qutip.ui.progressbar import progress_bars
from qutip.settings import available_cpu_count

if sys.platform == 'darwin':
    mp_context = multiprocessing.get_context('fork')
elif sys.platform == 'linux':
    # forkserver would handle threads better, but is much slower at starting
    # the executor and spawning tasks
    mp_context = multiprocessing.get_context('fork')
else:
    mp_context = multiprocessing.get_context()


default_map_kw = {
    'job_timeout': threading.TIMEOUT_MAX,
    'timeout': threading.TIMEOUT_MAX,
    'num_cpus': available_cpu_count(),
    'fail_fast': True,
}


def _read_map_kw(options):
    options = options or {}
    map_kw = default_map_kw.copy()
    map_kw.update({k: v for k, v in options.items() if v is not None})
    return map_kw


class MapExceptions(Exception):
    def __init__(self, msg, errors, results):
        super().__init__(msg, errors, results)
        self.errors = errors
        self.results = results


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
        storing a them in a list. It should return None or a number.
        When returning a number, it represent the estimation of the number of
        task left. On a return <= 0, the map will end early.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing:
        - timeout: float, Maximum time (sec) for the whole map.
        - fail_fast: bool, Raise an error at the first.

    Returns
    -------
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
    remaining_ntraj = None
    progress_bar = progress_bars[progress_bar](
        len(values), **progress_bar_kwargs
    )
    end_time = map_kw['timeout'] + time.time()
    results = None
    if reduce_func is None:
        results = [None] * len(values)
    errors = {}
    for n, value in enumerate(values):
        if time.time() > end_time:
            break
        progress_bar.update()
        try:
            result = task(value, *task_args, **task_kwargs)
        except Exception as err:
            if map_kw["fail_fast"]:
                raise err
            else:
                errors[n] = err
        else:
            if reduce_func is not None:
                remaining_ntraj = reduce_func(result)
            else:
                results[n] = result
        if remaining_ntraj is not None and remaining_ntraj <= 0:
            end_time = 0
    progress_bar.finished()

    if errors:
        raise MapExceptions(f"{len(errors)} iterations failed in serial_map",
                            errors, results)
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
        storing a them in a list. Note that the order in which results are
        passed to ``reduce_func`` is not defined. It should return None or a
        number. When returning a number, it represent the estimation of the
        number of task left. On a return <= 0, the map will end early.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for:
        - timeout: float, Maximum time (sec) for the whole map.
        - job_timeout: float, Maximum time (sec) for each job in the map.
        - num_cpus: int, Number of job to run at once.
        - fail_fast: bool, Raise an error at the first.

    Returns
    -------
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
    end_time = map_kw['timeout'] + time.time()
    job_time = map_kw['job_timeout']

    progress_bar = progress_bars[progress_bar](
        len(values), **progress_bar_kwargs
    )

    errors = {}
    finished = []
    if reduce_func is not None:
        results = None
        result_func = lambda i, value: reduce_func(value)
    else:
        results = [None] * len(values)
        result_func = lambda i, value: results.__setitem__(i, value)

    def _done_callback(future):
        if not future.cancelled():
            try:
                result = future.result()
            except Exception as e:
                errors[future._i] = e
        remaining_ntraj = result_func(future._i, result)
        if remaining_ntraj is not None and remaining_ntraj <= 0:
            finished.append(True)
        progress_bar.update()

    if sys.version_info >= (3, 7):
        # ProcessPoolExecutor only supports mp_context from 3.7 onwards
        ctx_kw = {"mp_context": mp_context}
    else:
        ctx_kw = {}

    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=map_kw['num_cpus'], **ctx_kw,
        ) as executor:
            waiting = set()
            i = 0
            while i < len(values):
                # feed values to the executor, ensuring that there is at
                # most one task per worker at any moment in time so that
                # we can shutdown without waiting for greater than the time
                # taken by the longest task
                if len(waiting) >= map_kw['num_cpus']:
                    # no space left, wait for a task to complete or
                    # the time to run out
                    timeout = max(0, end_time - time.time())
                    _done, waiting = concurrent.futures.wait(
                        waiting,
                        timeout=timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                if (
                    time.time() >= end_time
                    or (errors and map_kw['fail_fast'])
                    or finished
                ):
                    # no time left, exit the loop
                    break
                while len(waiting) < map_kw['num_cpus'] and i < len(values):
                    # space and time available, add tasks
                    value = values[i]
                    future = executor.submit(
                        task, *((value,) + task_args), **task_kwargs,
                    )
                    # small hack to avoid add_done_callback not supporting
                    # extra arguments and closures inside loops retaining
                    # a reference not a value:
                    future._i = i
                    future.add_done_callback(_done_callback)
                    waiting.add(future)
                    i += 1

            timeout = max(0, end_time - time.time())
            concurrent.futures.wait(waiting, timeout=timeout)
    finally:
        os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'

    progress_bar.finished()
    if errors and map_kw["fail_fast"]:
        raise list(errors.values())[0]
    elif errors:
        raise MapExceptions(
            f"{len(errors)} iterations failed in parallel_map",
            errors, results
        )

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
        storing a them in a list. It should return None or a number.  When
        returning a number, it represent the estimation of the number of task
        left. On a return <= 0, the map will end early.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for:
        - timeout: float, Maximum time (sec) for the whole map.
        - job_timeout: float, Maximum time (sec) for each job in the map.
        - num_cpus: int, Number of job to run at once.
        - fail_fast: bool, Raise an error at the first.

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

    progress_bar = progress_bars[progress_bar](
        len(values), **progress_bar_kwargs
    )

    executor = get_reusable_executor(max_workers=map_kw['num_cpus'])
    end_time = map_kw['timeout'] + time.time()
    job_time = map_kw['job_timeout']
    results = None
    remaining_ntraj = None
    errors = {}
    if reduce_func is None:
        results = [None] * len(values)

    try:
        jobs = [executor.submit(task, value, *task_args, **task_kwargs)
               for value in values]

        for n, job in enumerate(jobs):
            remaining_time = min(end_time - time.time(), job_time)
            try:
                result = job.result(remaining_time)
            except Exception as err:
                if map_kw["fail_fast"]:
                    raise err
                else:
                    errors[n] = err
            else:
                if reduce_func is not None:
                    remaining_ntraj = reduce_func(result)
                else:
                    results[n] = result
            progress_bar.update()
            if remaining_ntraj is not None and remaining_ntraj <= 0:
                break

    except KeyboardInterrupt as e:
        [job.cancel() for job in jobs]
        raise e

    except TimeoutError:
        [job.cancel() for job in jobs]

    finally:
        executor.shutdown()
    progress_bar.finished()
    os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
    if errors:
        raise MapExceptions(
            f"{len(errors)} iterations failed in loky_pmap",
            errors, results
        )
    return results


_get_map = {
    "parallel_map": parallel_map,
    "parallel": parallel_map,
    "serial_map": serial_map,
    "serial": serial_map,
    "loky": loky_pmap,
}

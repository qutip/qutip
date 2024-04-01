"""
This module provides functions for parallel execution of loops and function
mappings, using the builtin Python module multiprocessing or the loky parallel
execution library.
"""
__all__ = ['parallel_map', 'serial_map', 'loky_pmap', 'mpi_pmap']

import multiprocessing
import os
import sys
import time
import threading
import concurrent.futures
import warnings
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

    This function work as a drop-in replacement of :func:`parallel_map`.

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list, optional
        The optional additional argument to the ``task`` function.
    task_kwargs : dictionary, optional
        The optional additional keyword argument to the ``task`` function.
    reduce_func : func, optional
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list. It should return None or a number.
        When returning a number, it represent the estimation of the number of
        task left. On a return <= 0, the map will end early.
    progress_bar : str, optional
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict, optional
        Options for the progress bar.
    map_kw: dict, optional
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


def _generic_pmap(task, values, task_args, task_kwargs, reduce_func,
                  timeout, fail_fast, num_workers,
                  progress_bar, progress_bar_kwargs,
                  setup_executor, extract_result, shutdown_executor):
    """
    Common functionality for parallel_map, loky_pmap and mpi_pmap.
    The parameters `setup_executor`, `extract_result` and `shutdown_executor`
    are callback functions with the following signatures:

    setup_executor: () -> ProcessPoolExecutor

    extract_result: Future -> (Any, BaseException)
        If there was an exception e, returns (None, e).
        Otherwise returns (result, None).

    shutdown_executor: (executor: ProcessPoolExecutor,
                        active_tasks: set[Future]) -> None
        executor: The ProcessPoolExecutor that was created in setup_executor
        active_tasks: A set of Futures that are currently still being executed
            (non-empty if: timeout, error, or reduce_func requesting exit)
    """

    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    end_time = timeout + time.time()

    progress_bar = progress_bars[progress_bar](
        len(values), **progress_bar_kwargs
    )

    errors = {}
    finished = []
    if reduce_func is not None:
        results = None

        def result_func(_, value):
            return reduce_func(value)
    else:
        results = [None] * len(values)
        result_func = results.__setitem__

    def _done_callback(future):
        if not future.cancelled():
            result, exception = extract_result(future)
            if isinstance(exception, KeyboardInterrupt):
                # When a keyboard interrupt happens, it is raised in the main
                # thread and in all worker threads. At this point in the code,
                # the worker threads have already returned and the main thread
                # is only waiting for the ProcessPoolExecutor to shutdown
                # before exiting. We therefore return immediately.
                return
            if exception is not None:
                if isinstance(exception, Exception):
                    errors[future._i] = exception
                else:
                    raise exception
            else:
                remaining_ntraj = result_func(future._i, result)
                if remaining_ntraj is not None and remaining_ntraj <= 0:
                    finished.append(True)
        progress_bar.update()

    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    try:
        with setup_executor() as executor:
            waiting = set()
            i = 0
            aborted = False

            while i < len(values):
                # feed values to the executor, ensuring that there is at
                # most one task per worker at any moment in time so that
                # we can shutdown without waiting for greater than the time
                # taken by the longest task
                if len(waiting) >= num_workers:
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
                    or (errors and fail_fast)
                    or finished
                ):
                    # no time left, exit the loop
                    aborted = True
                    break
                while len(waiting) < num_workers and i < len(values):
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

            if not aborted:
                # all tasks have been submitted, timeout has not been reaches
                # -> wait for all workers to finish before shutting down
                timeout = max(0, end_time - time.time())
                _done, waiting = concurrent.futures.wait(
                    waiting,
                    timeout=timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
            shutdown_executor(executor, waiting)
    finally:
        os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'

    progress_bar.finished()
    if errors and fail_fast:
        raise list(errors.values())[0]
    elif errors:
        raise MapExceptions(
            f"{len(errors)} iterations failed in parallel_map",
            errors, results
        )

    return results


def parallel_map(task, values, task_args=None, task_kwargs=None,
                 reduce_func=None, map_kw=None,
                 progress_bar=None, progress_bar_kwargs={}):
    """
    Parallel execution of a mapping of ``values`` to the function ``task``.
    This is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list, optional
        The optional additional arguments to the ``task`` function.
    task_kwargs : dictionary, optional
        The optional additional keyword arguments to the ``task`` function.
    reduce_func : func, optional
        If provided, it will be called with the output of each task instead of
        storing them in a list. Note that the order in which results are
        passed to ``reduce_func`` is not defined. It should return None or a
        number. When returning a number, it represents the estimation of the
        number of tasks left. On a return <= 0, the map will end early.
    progress_bar : str, optional
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict, optional
        Options for the progress bar.
    map_kw: dict, optional
        Dictionary containing entry for:
        - timeout: float, Maximum time (sec) for the whole map.
        - num_cpus: int, Number of jobs to run at once.
        - fail_fast: bool, Abort at the first error.

    Returns
    -------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """

    map_kw = _read_map_kw(map_kw)
    if sys.version_info >= (3, 7):
        # ProcessPoolExecutor only supports mp_context from 3.7 onwards
        ctx_kw = {"mp_context": mp_context}
    else:
        ctx_kw = {}

    def setup_executor():
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=map_kw['num_cpus'], **ctx_kw,
        )

    def extract_result(future: concurrent.futures.Future):
        exception = future.exception()
        if exception is not None:
            return None, exception
        return future.result(), None

    def shutdown_executor(executor, _):
        # Since `ProcessPoolExecutor` leaves no other option,
        # we wait for all worker processes to finish their current task
        executor.shutdown()

    return _generic_pmap(
        task, values, task_args, task_kwargs, reduce_func,
        map_kw['timeout'], map_kw['fail_fast'], map_kw['num_cpus'],
        progress_bar, progress_bar_kwargs,
        setup_executor, extract_result, shutdown_executor
    )


def loky_pmap(task, values, task_args=None, task_kwargs=None,
              reduce_func=None, map_kw=None,
              progress_bar=None, progress_bar_kwargs={}):
    """
    Parallel execution of a mapping of ``values`` to the function ``task``.
    This is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Use the loky module instead of multiprocessing.

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list, optional
        The optional additional arguments to the ``task`` function.
    task_kwargs : dictionary, optional
        The optional additional keyword arguments to the ``task`` function.
    reduce_func : func, optional
        If provided, it will be called with the output of each task instead of
        storing them in a list. Note that the order in which results are
        passed to ``reduce_func`` is not defined. It should return None or a
        number. When returning a number, it represents the estimation of the
        number of tasks left. On a return <= 0, the map will end early.
    progress_bar : str, optional
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict, optional
        Options for the progress bar.
    map_kw: dict, optional
        Dictionary containing entry for:
        - timeout: float, Maximum time (sec) for the whole map.
        - num_cpus: int, Number of jobs to run at once.
        - fail_fast: bool, Abort at the first error.

    Returns
    -------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """

    from loky import get_reusable_executor
    from loky.process_executor import ShutdownExecutorError
    map_kw = _read_map_kw(map_kw)

    def setup_executor():
        return get_reusable_executor(max_workers=map_kw['num_cpus'])

    def extract_result(future: concurrent.futures.Future):
        exception = future.exception()
        if isinstance(exception, ShutdownExecutorError):
            # Task was aborted due to timeout etc
            return None, None
        if exception is not None:
            return None, exception
        return future.result(), None

    def shutdown_executor(executor, active_tasks):
        # If there are still tasks running, we kill all workers in  order to
        # return immediately. Otherwise, `kill_workers` is set to False so
        # that the worker threads can be reused in subsequent loky_pmap calls.
        kill_workers = len(active_tasks) > 0
        executor.shutdown(kill_workers=kill_workers)

    return _generic_pmap(
        task, values, task_args, task_kwargs, reduce_func,
        map_kw['timeout'], map_kw['fail_fast'], map_kw['num_cpus'],
        progress_bar, progress_bar_kwargs,
        setup_executor, extract_result, shutdown_executor
    )


def mpi_pmap(task, values, task_args=None, task_kwargs=None,
             reduce_func=None, map_kw=None,
             progress_bar=None, progress_bar_kwargs={}):
    """
    Parallel execution of a mapping of ``values`` to the function ``task``.
    This is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Uses the mpi4py module to execute the tasks asynchronously with MPI
    processes. For more information, consult the documentation of mpi4py and
    the mpi4py.MPIPoolExecutor class.

    Note: in keeping consistent with the API of `parallel_map`, the parameter
    determining the number of requested worker processes is called `num_cpus`.
    The value of `map_kw['num_cpus']` is passed to the MPIPoolExecutor as its
    `max_workers` argument.
    If this parameter is not provided, the environment variable
    `QUTIP_NUM_PROCESSES` is used instead. If this environment variable is not
    set either, QuTiP will use default values that might be unsuitable for MPI
    applications.

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list, optional
        The optional additional arguments to the ``task`` function.
    task_kwargs : dictionary, optional
        The optional additional keyword arguments to the ``task`` function.
    reduce_func : func, optional
        If provided, it will be called with the output of each task instead of
        storing them in a list. Note that the order in which results are
        passed to ``reduce_func`` is not defined. It should return None or a
        number. When returning a number, it represents the estimation of the
        number of tasks left. On a return <= 0, the map will end early.
    progress_bar : str, optional
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict, optional
        Options for the progress bar.
    map_kw: dict, optional
        Dictionary containing entry for:
        - timeout: float, Maximum time (sec) for the whole map.
        - num_cpus: int, Number of jobs to run at once.
        - fail_fast: bool, Abort at the first error.
        All remaining entries of map_kw will be passed to the
        mpi4py.MPIPoolExecutor constructor.

    Returns
    -------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """

    from mpi4py.futures import MPIPoolExecutor

    # If the provided num_cpus is None, we use the default value instead.
    # We thus intentionally make it impossible to call
    #   MPIPoolExecutor(max_workers=None, ...)
    # in which case mpi4py would determine a default value. That would be
    # useful, but unfortunately mpi4py provides no public API to access the
    # actual number of workers that is used in that case, which we would need.
    worker_number_provided = (
        ((map_kw is not None) and ('num_cpus' in map_kw))
        or 'QUTIP_NUM_PROCESSES' in os.environ)

    map_kw = _read_map_kw(map_kw)
    timeout = map_kw.pop('timeout')
    num_workers = map_kw.pop('num_cpus')
    fail_fast = map_kw.pop('fail_fast')

    if not worker_number_provided:
        warnings.warn(f'mpi_pmap was called without specifying the number of '
                      f'worker processes, using the default {num_workers}')

    def setup_executor():
        return MPIPoolExecutor(max_workers=num_workers, **map_kw)

    def extract_result(future):
        exception = future.exception()
        if exception is not None:
            return None, exception
        return future.result(), None

    def shutdown_executor(executor, _):
        executor.shutdown()

    return _generic_pmap(
        task, values, task_args, task_kwargs, reduce_func,
        timeout, fail_fast, num_workers,
        progress_bar, progress_bar_kwargs,
        setup_executor, extract_result, shutdown_executor
    )


_maps = {
    "parallel_map": parallel_map,
    "parallel": parallel_map,
    "serial_map": serial_map,
    "serial": serial_map,
    "loky": loky_pmap,
    "mpi": mpi_pmap
}


def _get_map(options):
    map_func = _maps[options['map']]

    if map_func == mpi_pmap:
        map_kw = options['mpi_options']
    else:
        map_kw = {}

    return map_func, map_kw

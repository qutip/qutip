"""
This module contains utility functions for using QuTiP with IPython notebooks.
"""
from qutip.ui.progressbar import BaseProgressBar, HTMLProgressBar
from .settings import _blas_info, available_cpu_count
import IPython

#IPython parallel routines moved to ipyparallel in V4
#IPython parallel routines not in Anaconda by default
if IPython.version_info[0] >= 4:
    try:
        from ipyparallel import Client
        __all__ = ['version_table', 'plot_animation',
                    'parallel_map']
    except:
         __all__ = ['version_table', 'plot_animation']
else:
    try:
        from IPython.parallel import Client
        __all__ = ['version_table', 'plot_animation',
                    'parallel_map']
    except:
         __all__ = ['version_table', 'plot_animation']


from IPython.display import HTML, Javascript, display

import matplotlib.pyplot as plt
from matplotlib import animation
from base64 import b64encode

import datetime
import uuid
import sys
import os
import time
import inspect

import qutip
import numpy
import scipy
import matplotlib
import IPython

try:
    import Cython
    _cython_available = True
except ImportError:
    _cython_available = False


def version_table(verbose=False):
    """
    Print an HTML-formatted table with version numbers for QuTiP and its
    dependencies. Use it in a IPython notebook to show which versions of
    different packages that were used to run the notebook. This should make it
    possible to reproduce the environment and the calculation later on.

    Parameters
    ----------
    verbose : bool, default: False
        Add extra information about install location.

    Returns
    -------
    version_table: str
        Return an HTML-formatted string containing version information for
        QuTiP dependencies.

    """

    html = "<table>"
    html += "<tr><th>Software</th><th>Version</th></tr>"

    packages = [("QuTiP", qutip.__version__),
                ("Numpy", numpy.__version__),
                ("SciPy", scipy.__version__),
                ("matplotlib", matplotlib.__version__),
                ("Number of CPUs", available_cpu_count()),
                ("BLAS Info", _blas_info()),
                ("IPython", IPython.__version__),
                ("Python", sys.version),
                ("OS", "%s [%s]" % (os.name, sys.platform))
                ]
    if _cython_available:
        packages.append(("Cython", Cython.__version__))

    for name, version in packages:
        html += "<tr><td>%s</td><td>%s</td></tr>" % (name, version)

    if verbose:
        html += "<tr><th colspan='2'>Additional information</th></tr>"
        qutip_install_path = os.path.dirname(inspect.getsourcefile(qutip))
        html += ("<tr><td>Installation path</td><td>%s</td></tr>" %
                 qutip_install_path)
        try:
            import getpass
            html += ("<tr><td>User</td><td>%s</td></tr>" %
                     getpass.getuser())
        except:
            pass

    html += "<tr><td colspan='2'>%s</td></tr>" % time.strftime(
        '%a %b %d %H:%M:%S %Y %Z')
    html += "</table>"

    return HTML(html)


def _visualize_parfor_data(metadata):
    """
    Visualizing the task scheduling meta data collected from AsyncResults.
    """
    res = numpy.array(metadata)
    fig, ax = plt.subplots(figsize=(10, res.shape[1]))

    yticks = []
    yticklabels = []
    tmin = min(res[:, 1])
    for n, pid in enumerate(numpy.unique(res[:, 0])):
        yticks.append(n)
        yticklabels.append("%d" % pid)
        for m in numpy.where(res[:, 0] == pid)[0]:
            ax.add_patch(plt.Rectangle((res[m, 1] - tmin, n - 0.25),
                         res[m, 2] - res[m, 1], 0.5, color="green", alpha=0.5))

    ax.set_ylim(-.5, n + .5)
    ax.set_xlim(0, max(res[:, 2]) - tmin + 0.)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Engine")
    ax.set_xlabel("seconds")
    ax.set_title("Task schedule")


def parfor(task, task_vec, args=None, client=None, view=None,
           show_scheduling=False, show_progressbar=False):
    """
    Call the function ``tast`` for each value in ``task_vec`` using a cluster
    of IPython engines. The function ``task`` should have the signature
    ``task(value, args)`` or ``task(value)`` if ``args=None``.

    The ``client`` and ``view`` are the IPython.parallel client and
    load-balanced view that will be used in the parfor execution. If these
    are ``None``, new instances will be created.

    Parameters
    ----------

    task: a Python function
        The function that is to be called for each value in ``task_vec``.

    task_vec: array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.

    args: list / dictionary
        The optional additional argument to the ``task`` function. For example
        a dictionary with parameter values.

    client: IPython.parallel.Client
        The IPython.parallel Client instance that will be used in the
        parfor execution.

    view: a IPython.parallel.Client view
        The view that is to be used in scheduling the tasks on the IPython
        cluster. Preferably a load-balanced view, which is obtained from the
        IPython.parallel.Client instance client by calling,
        view = client.load_balanced_view().

    show_scheduling: bool {False, True}, default False
        Display a graph showing how the tasks (the evaluation of ``task`` for
        for the value in ``task_vec1``) was scheduled on the IPython engine
        cluster.

    show_progressbar: bool {False, True}, default False
        Display a HTML-based progress bar duing the execution of the parfor
        loop.

    Returns
    -------
    result : list
        The result list contains the value of ``task(value, args)`` for each
        value in ``task_vec``, that is, it should be equivalent to
        ``[task(v, args) for v in task_vec]``.
    """

    if show_progressbar:
        progress_bar = HTMLProgressBar()
    else:
        progress_bar = None

    return parallel_map(task, task_vec, task_args=args,
                        client=client, view=view, progress_bar=progress_bar,
                        show_scheduling=show_scheduling)


def parallel_map(task, values, task_args=None, task_kwargs=None,
                 client=None, view=None, progress_bar=None,
                 show_scheduling=False, **kwargs):
    """
    Call the function ``task`` for each value in ``values`` using a cluster
    of IPython engines. The function ``task`` should have the signature
    ``task(value, *args, **kwargs)``.

    The ``client`` and ``view`` are the IPython.parallel client and
    load-balanced view that will be used in the parfor execution. If these
    are ``None``, new instances will be created.

    Parameters
    ----------

    task: a Python function
        The function that is to be called for each value in ``task_vec``.

    values: array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.

    task_args: list / dictionary
        The optional additional argument to the ``task`` function.

    task_kwargs: list / dictionary
        The optional additional keyword argument to the ``task`` function.

    client: IPython.parallel.Client
        The IPython.parallel Client instance that will be used in the
        parfor execution.

    view: a IPython.parallel.Client view
        The view that is to be used in scheduling the tasks on the IPython
        cluster. Preferably a load-balanced view, which is obtained from the
        IPython.parallel.Client instance client by calling,
        view = client.load_balanced_view().

    show_scheduling: bool {False, True}, default False
        Display a graph showing how the tasks (the evaluation of ``task`` for
        for the value in ``task_vec1``) was scheduled on the IPython engine
        cluster.

    show_progressbar: bool {False, True}, default False
        Display a HTML-based progress bar during the execution of the parfor
        loop.

    Returns
    -------
    result : list
        The result list contains the value of
        ``task(value, task_args, task_kwargs)`` for each
        value in ``values``.

    """
    submitted = datetime.datetime.now()

    if task_args is None:
        task_args = tuple()

    if task_kwargs is None:
        task_kwargs = {}

    if client is None:
        client = Client()

        # make sure qutip is available at engines
        dview = client[:]
        dview.block = True
        dview.execute("from qutip import *")

    if view is None:
        view = client.load_balanced_view()

    ar_list = [view.apply_async(task, value, *task_args, **task_kwargs)
               for value in values]

    if progress_bar is None:
        view.wait(ar_list)
    else:
        if progress_bar is True:
            progress_bar = HTMLProgressBar(len(ar_list))
        prev_finished = 0
        while True:
            n_finished = sum([ar.progress for ar in ar_list])
            for _ in range(prev_finished, n_finished):
                progress_bar.update()
            prev_finished = n_finished

            if view.wait(ar_list, timeout=0.5):
                break
        progress_bar.finished()

    if show_scheduling:
        metadata = [[ar.engine_id,
                     (ar.started - submitted).total_seconds(),
                     (ar.completed - submitted).total_seconds()]
                    for ar in ar_list]
        _visualize_parfor_data(metadata)

    return [ar.get() for ar in ar_list]


def plot_animation(plot_setup_func, plot_func, result, name="movie",
                   writer="avconv", codec="libx264", verbose=False):
    """
    Create an animated plot of a Result object, as returned by one of
    the qutip evolution solvers.

    .. note :: experimental
    """

    fig, axes = plot_setup_func(result)

    def update(n):
        return plot_func(result, n, fig=fig, axes=axes)

    anim = animation.FuncAnimation(
        fig, update, frames=len(result.times), blit=True)

    anim.save(name + '.mp4', fps=10, writer=writer, codec=codec)

    plt.close(fig)

    if verbose:
        print("Created %s.m4v" % name)

    video = open(name + '.mp4', "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = '<video controls src="data:video/x-m4v;base64,{0}">'.format(
        video_encoded)
    return HTML(video_tag)

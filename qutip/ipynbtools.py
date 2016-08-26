# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module contains utility functions for using QuTiP with IPython notebooks.
"""
from qutip.ui.progressbar import BaseProgressBar
from qutip.utilities import _blas_info
import IPython

#IPython parallel routines moved to ipyparallel in V4
#IPython parallel routines not in Anaconda by default
if IPython.version_info[0] >= 4:
    try:
        from ipyparallel import Client
        __all__ = ['version_table', 'parfor', 'plot_animation', 
                    'parallel_map', 'HTMLProgressBar']
    except:
         __all__ = ['version_table', 'plot_animation', 'HTMLProgressBar']
else:
    try:
        from IPython.parallel import Client
        __all__ = ['version_table', 'parfor', 'plot_animation', 
                    'parallel_map', 'HTMLProgressBar']
    except:
         __all__ = ['version_table', 'plot_animation', 'HTMLProgressBar']


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
import Cython
import matplotlib
import IPython


def version_table(verbose=False):
    """
    Print an HTML-formatted table with version numbers for QuTiP and its
    dependencies. Use it in a IPython notebook to show which versions of
    different packages that were used to run the notebook. This should make it
    possible to reproduce the environment and the calculation later on.


    Returns
    --------
    version_table: string
        Return an HTML-formatted string containing version information for
        QuTiP dependencies.

    """

    html = "<table>"
    html += "<tr><th>Software</th><th>Version</th></tr>"

    packages = [("QuTiP", qutip.__version__),
                ("Numpy", numpy.__version__),
                ("SciPy", scipy.__version__),
                ("matplotlib", matplotlib.__version__),
                ("Cython", Cython.__version__),
                ("Number of CPUs", qutip.hardware_info.hardware_info()['cpus']),
                ("BLAS Info", _blas_info()),
                ("IPython", IPython.__version__),
                ("Python", sys.version),
                ("OS", "%s [%s]" % (os.name, sys.platform))
                ]

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


class HTMLProgressBar(BaseProgressBar):
    """
    A simple HTML progress bar for using in IPython notebooks. Based on
    IPython ProgressBar demo notebook:
    https://github.com/ipython/ipython/tree/master/examples/notebooks

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = HTMLProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update(n)
            compute_with_n(n)
    """

    def __init__(self, iterations=0, chunk_size=1.0):
        self.divid = str(uuid.uuid4())
        self.textid = str(uuid.uuid4())
        self.pb = HTML("""\
<div style="border: 2px solid grey; width: 600px">
  <div id="%s" \
style="background-color: rgba(121,195,106,0.75); width:0%%">&nbsp;</div>
</div>
<p id="%s"></p>
""" % (self.divid, self.textid))
        display(self.pb)
        super(HTMLProgressBar, self).start(iterations, chunk_size)

    def start(self, iterations=0, chunk_size=1.0):
        super(HTMLProgressBar, self).start(iterations, chunk_size)

    def update(self, n):
        p = (n / self.N) * 100.0
        if p >= self.p_chunk:
            lbl = ("Elapsed time: %s. " % self.time_elapsed() +
                   "Est. remaining time: %s." % self.time_remaining_est(p))
            js_code = ("$('div#%s').width('%i%%');" % (self.divid, p) +
                       "$('p#%s').text('%s');" % (self.textid, lbl))
            display(Javascript(js_code))
            # display(Javascript("$('div#%s').width('%i%%')" % (self.divid,
            # p)))
            self.p_chunk += self.p_chunk_size

    def finished(self):
        self.t_done = time.time()
        lbl = "Elapsed time: %s" % self.time_elapsed()
        js_code = ("$('div#%s').width('%i%%');" % (self.divid, 100.0) +
                   "$('p#%s').text('%s');" % (self.textid, lbl))
        display(Javascript(js_code))


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
    --------
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
    --------
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
            progress_bar = HTMLProgressBar()

        n = len(ar_list)
        progress_bar.start(n)
        while True:
            n_finished = sum([ar.progress for ar in ar_list])
            progress_bar.update(n_finished)

            if view.wait(ar_list, timeout=0.5):
                progress_bar.update(n)
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
        plot_func(result, n, fig=fig, axes=axes)

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

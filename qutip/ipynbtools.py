# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################
"""
This module contains utility functions for using QuTiP with IPython notebooks.
"""

from IPython.parallel import Client
from IPython.display import HTML, Javascript, display

import matplotlib.pyplot as plt

import datetime
import uuid
import sys
import os
import time

import qutip
import numpy
import scipy
import Cython
import matplotlib
import IPython


def version_table():
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

    packages = {"QuTiP": qutip.__version__,
                "Numpy": numpy.__version__,
                "SciPy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
                "Cython": Cython.__version__,
                "Python": sys.version,
                "IPython": IPython.__version__,
                "OS": "%s [%s]" % (os.name, sys.platform)
                }

    for name in packages:
        html += "<tr><td>%s</td><td>%s</td></tr>" % (name, packages[name])

    html += "<tr><td colspan='2'>%s</td></tr>" % time.ctime()
    html += "</table>"

    return HTML(html)


class _HTMLProgressBar():
    """
    Based on IPython ProgressBar demo notebook:
    https://github.com/ipython/ipython/tree/master/examples/notebooks
    """

    def __init__(self, iterations):
        self.N = float(iterations)
        self.divid = str(uuid.uuid4())
        self.pb = HTML("""\
<div style="border: 1px solid grey; width: 600px">
  <div id="%s" \
style="background-color: rgba(0,200,0,0.35); width:0%%">&nbsp;</div>
</div>
""" % self.divid)
        display(self.pb)

    def update(self, n):
        p = (n / self.N) * 100.0
        display(Javascript("$('div#%s').width('%i%%')" % (self.divid, p)))


def _visualize_parfor_data(metadata):
    """
    Visalizing the task scheduling meta data collected from AsyncResults.
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

    submitted = datetime.datetime.now()

    if client is None:
        client = Client()

        # make sure qutip is available at engines
        dview = client[:]
        dview.block = True
        dview.execute("from qutip import *")

    if view is None:
        view = client.load_balanced_view()

    if args is None:
        ar_list = [view.apply_async(task, x) for x in task_vec]
    else:
        ar_list = [view.apply_async(task, x, args) for x in task_vec]

    if show_progressbar:
        n = len(ar_list)
        pbar = _HTMLProgressBar(n)
        while True:
            n_finished = sum([ar.progress for ar in ar_list])
            pbar.update(n_finished)

            if view.wait(ar_list, timeout=0.5):
                pbar.update(n)
                break
    else:
        view.wait(ar_list)

    if show_scheduling:
        metadata = [[ar.engine_id,
                     (ar.started - submitted).total_seconds(),
                     (ar.completed - submitted).total_seconds()]
                    for ar in ar_list]
        _visualize_parfor_data(metadata)

    return [ar.get() for ar in ar_list]

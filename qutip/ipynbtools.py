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

from IPython.core.display import HTML
from IPython.parallel import Client

import sys
import os
import qutip
import numpy
import scipy
import Cython
import matplotlib
import IPython
import time


def version_table():
    """
    Print an HTML-formatted table with version numbers for QuTiP and its
    dependencies. Use it in a IPython notebook to show which versions of
    different packages that were used to run the notebook. This should make it
    possible to reproduce the environment and the calculation later on.
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


def ipy_parfor(task, task_vec, args=None, client=None, view=None):
    """
    Call the function 'tast' for each value in 'task_vec' using a cluster
    of IPython engines.
    """

    if client is None:
        client = Client()

    if view is None:
        view = client.load_balanced_view()

    if args is None:
        ar = [view.apply_async(task, x) for x in task_vec]
    else:
        ar = [view.apply_async(task, x, args) for x in task_vec]

    view.wait(ar)

    return [a.get() for a in ar]


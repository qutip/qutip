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

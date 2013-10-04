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
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
Command line output of information on QuTiP and
dependencies.
"""

import sys
import os
import numpy
import scipy
import qutip.settings
from qutip import __version__ as qutip_version


def about():
    """
    About box for qutip. Gives version numbers for
    QuTiP, NumPy, SciPy, Cython, and MatPlotLib.
    """
    print('')
    print("QuTiP: The Quantum Toolbox in Python")
    print("Copyright (c) 2011 and later.")
    print("Paul D. Nation & Robert J. Johansson")
    print('')
    print("QuTiP Version:       " + qutip.__version__)
    print("Numpy Version:       " + numpy.__version__)
    print("Scipy Version:       " + scipy.__version__)
    try:
        import Cython
        cython_ver = Cython.__version__
    except:
        cython_ver = 'None'
    print(("Cython Version:      " + cython_ver))
    try:
        import matplotlib
        matplotlib_ver = matplotlib.__version__
    except:
        matplotlib_ver = 'None'
    print(("Matplotlib Version:  " + matplotlib_ver))
    print('')

if __name__ == "__main__":
    about()

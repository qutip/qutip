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
    print("QuTiP: Quantum Toolbox in Python")
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

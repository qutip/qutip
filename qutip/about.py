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
Command line output of information on QuTiP and dependencies.
"""
__all__ = ['about', 'cite']

import sys
import os
import platform
import numpy
import scipy
import inspect
from qutip.utilities import _blas_info
import qutip.settings
from qutip.hardware_info import hardware_info


def about():
    """
    About box for QuTiP. Gives version numbers for
    QuTiP, NumPy, SciPy, Cython, and MatPlotLib.
    """
    print("")
    print("QuTiP: Quantum Toolbox in Python")
    print("Copyright (c) 2011 and later.")
    print(("A. J. Pitchford, P. D. Nation, "
            "R. J. Johansson, A. Grimsmo, "
            "and C. Granade"))
    print("")
    print("QuTiP Version:      %s" % qutip.__version__)
    print("Numpy Version:      %s" % numpy.__version__)
    print("Scipy Version:      %s" % scipy.__version__)
    try:
        import Cython
        cython_ver = Cython.__version__
    except:
        cython_ver = 'None'
    print("Cython Version:     %s" % cython_ver)
    try:
        import matplotlib
        matplotlib_ver = matplotlib.__version__
    except:
        matplotlib_ver = 'None'
    print("Matplotlib Version: %s" % matplotlib_ver)
    print("Python Version:     %d.%d.%d" % sys.version_info[0:3])
    print("Number of CPUs:     %s" % hardware_info()['cpus'])
    print("BLAS Info:          %s" % _blas_info())
    print("OPENMP Installed:   %s" % str(qutip.settings.has_openmp))
    print("INTEL MKL Ext:      %s" % str(qutip.settings.has_mkl))
    print("Platform Info:      %s (%s)" % (platform.system(),
                                           platform.machine()))
    qutip_install_path = os.path.dirname(inspect.getsourcefile(qutip))
    print("Installation path:  %s" % qutip_install_path)
    print("Please cite QuTiP in your publication.")
    print("For your convenience a bibtex file can be easily generated using `qutip.about.cite()`")

def cite(path=None, verbose=True):
    """
    Citation information and bibtex generator for QuTiP

    Parameters
    ----------
    path: str
        The complete directory path to generate the bibtex file.
        If not specified then the citation will be generated in cwd
    """
    citation = ["@article{qutip2,",
                "doi = {10.1016/j.cpc.2012.11.019},",
                "url = {https://doi.org/10.1016/j.cpc.2012.11.019},",
                "year  = {2013},",
                "month = {apr},",
                "publisher = {Elsevier {BV}},",
                "volume = {184},",
                "number = {4},",
                "pages = {1234--1240},",
                "author = {J.R. Johansson and P.D. Nation and Franco Nori},",
                "title = {{QuTiP} 2: A Python framework for the dynamics of open quantum systems},",
                "journal = {Computer Physics Communications}",
                "}",
                "@article{qutip1,",
                "doi = {10.1016/j.cpc.2012.02.021},",
                "url = {https://doi.org/10.1016/j.cpc.2012.02.021},",
                "year  = {2012},",
                "month = {aug},",
                "publisher = {Elsevier {BV}},",
                "volume = {183},",
                "number = {8},",
                "pages = {1760--1772},",
                "author = {J.R. Johansson and P.D. Nation and Franco Nori},",
                "title = {{QuTiP}: An open-source Python framework for the dynamics of open quantum systems},",
                "journal = {Computer Physics Communications}",
                "}"]

    if verbose:
        print("\n".join(citation))

    if not path:
        path = os.getcwd()

    filename = "qutip.bib"
    with open(os.path.join(path, filename), 'w') as f:
        f.write("\n".join(citation))


if __name__ == "__main__":
    about()

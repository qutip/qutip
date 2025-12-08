"""
Command line output of information on QuTiP and dependencies.
"""
__all__ = ['about']

import sys
import os
import importlib.metadata
import platform
import numpy
import scipy
import inspect
import qutip
from qutip.settings import _blas_info, settings


def about():
    """
    About box for QuTiP. Gives version numbers for QuTiP, NumPy, SciPy, Cython,
    and MatPlotLib and information about installed QuTiP family packages.
    """
    print("")
    print("QuTiP: Quantum Toolbox in Python")
    print("================================")
    print("Copyright (c) QuTiP team 2011 and later.")
    print(
        "Current admin team: Alexander Pitchford, "
        "Nathan Shammah, Shahnawaz Ahmed, Neill Lambert, Eric Giguère, "
        "Boxi Li, Simon Cross, Asier Galicia, Paul Menczel, "
        "and Patrick Hopf."
    )
    print(
        "Board members: Daniel Burgarth, Robert Johansson, Anton F. Kockum, "
        "Franco Nori and Will Zeng."
    )
    print("Original developers: R. J. Johansson & P. D. Nation.")
    print("Previous lead developers: Chris Granade & A. Grimsmo.")
    print("Currently developed through wide collaboration. "
          "See https://github.com/qutip for details.")
    print("")
    print("QuTiP Version:      %s" % qutip.__version__)
    print("Numpy Version:      %s" % numpy.__version__)
    print("Scipy Version:      %s" % scipy.__version__)
    try:
        import Cython
        cython_ver = Cython.__version__
    except ImportError:
        cython_ver = 'None'
    print("Cython Version:     %s" % cython_ver)
    try:
        import matplotlib
        matplotlib_ver = matplotlib.__version__
    except ImportError:
        matplotlib_ver = 'None'
    print("Matplotlib Version: %s" % matplotlib_ver)
    print("Python Version:     %d.%d.%d" % sys.version_info[0:3])
    print("Number of CPUs:     %s" % settings.num_cpus)
    print("BLAS Info:          %s" % _blas_info())
    # print("OPENMP Installed:   %s" % str(qutip.settings.has_openmp))
    print("INTEL MKL Ext:      %s" % settings.mkl_lib_location)
    print("Platform Info:      %s (%s)" % (platform.system(),
                                           platform.machine()))
    qutip_install_path = os.path.dirname(inspect.getsourcefile(qutip))
    print("Installation path:  %s" % qutip_install_path)
    print()

    # family packages

    print("Installed QuTiP family packages")
    print("-------------------------------")
    print()

    entrypoints = importlib.metadata.entry_points(group="qutip.family")

    if not entrypoints:
        print("No QuTiP family packages installed.")

    for ep in entrypoints:
        family_mod = ep.load()
        try:
            pkg, version = family_mod.version()
        except Exception as exc:
            pkg, version = ep.name, [str(exc)]
        print("%s: %s" % (pkg, version))

    print()

    # citation
    longbar = "=" * 80
    cite_msg = "For your convenience a bibtex reference can be easily"
    cite_msg += " generated using `qutip.cite()`"
    print(longbar)
    print("Please cite QuTiP in your publication.")
    print(longbar)
    print(cite_msg)


if __name__ == "__main__":
    about()

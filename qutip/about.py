"""
Command line output of information on QuTiP and dependencies.
"""
__all__ = ['about']

import sys
import os
import platform
import numpy
import scipy
import inspect
from qutip.utilities import _blas_info, available_cpu_count
import qutip.settings


def about(caller="qutip"):
    """
    About box for QuTiP. Gives version numbers for QuTiP, NumPy, SciPy, Cython,
    and MatPlotLib.

    Parameters
    ----------
    caller: string
            The library about which the information is needed. Ex: 'qutip', 'qutip_qip' 
    """
    print("")
    print("QuTiP: Quantum Toolbox in Python")
    if caller == "qutip_qip":
        print("Quantum Information Processing Module")
    print("================================")
    print("Copyright (c) QuTiP team 2011 and later.")
    print(
        "Current admin team: Alexander Pitchford, "
        "Nathan Shammah, Shahnawaz Ahmed, Neill Lambert, Eric Giguère, "
        "Boxi Li, Jake Lishman and Simon Cross."
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
    if caller == "qutip_qip":
        try:
            import qutip_qip
            qutip_qip_ver = qutip_qip.__version__
        except ImportError:
            qutip_qip_ver = 'None'
        print("QuTiP QIP Version:  %s" % qutip_qip_ver)
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
    print("Number of CPUs:     %s" % available_cpu_count())
    print("BLAS Info:          %s" % _blas_info())
    print("OPENMP Installed:   %s" % str(qutip.settings.has_openmp))
    print("INTEL MKL Ext:      %s" % str(qutip.settings.has_mkl))
    print("Platform Info:      %s (%s)" % (platform.system(),
                                           platform.machine()))
    if caller == "qutip":
        install_path = os.path.dirname(inspect.getsourcefile(qutip))
    elif caller == "qutip_qip":
        try:
            install_path = os.path.dirname(inspect.getsourcefile(qutip_qip))
        except ImportError:
            pass
    print("Installation path:  %s" % install_path)
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

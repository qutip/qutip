"""
Command line output of information on QuTiP and dependencies.
"""
__all__ = ['about']

import sys
import importlib.metadata
import platform
from pathlib import Path
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
    print(f"QuTiP Version:      {qutip.__version__}")
    print(f"Numpy Version:      {numpy.__version__}")
    print(f"Scipy Version:      {scipy.__version__}")
    try:
        import Cython
        cython_ver = Cython.__version__
    except ImportError:
        cython_ver = 'None'
    print(f"Cython Version:     {cython_ver}")
    try:
        import matplotlib
        matplotlib_ver = matplotlib.__version__
    except ImportError:
        matplotlib_ver = 'None'
    print(f"Matplotlib Version: {matplotlib_ver}")
    print(f"Python Version:     {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")
    print(f"Number of CPUs:     {settings.num_cpus}")
    print(f"BLAS Info:          {_blas_info()}")
    # print(f"OPENMP Installed:   {str(qutip.settings.has_openmp)}")
    print(f"INTEL MKL Ext:      {settings.mkl_lib_location}")
    print(f"Platform Info:      {platform.system()} ({platform.machine()})")
    qutip_install_path = Path(inspect.getsourcefile(qutip)).parent
    print(f"Installation path:  {qutip_install_path}")
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
        print(f"{pkg}: {version}")

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

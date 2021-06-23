import os
import sys
from qutip.utilities import _blas_info
import qutip.settings as qset
from ctypes import cdll


def _set_mkl():
    """
    Finds the MKL runtime library for the
    Anaconda and Intel Python distributions.

    """
    if (
        _blas_info() != 'INTEL MKL'
        or sys.platform not in ['darwin', 'win32', 'linux', 'linux2']
    ):
        return
    python_dir = os.path.dirname(sys.executable)
    if sys.platform in ['darwin', 'linux2', 'linux']:
        python_dir = os.path.dirname(python_dir)
    library = {
        'darwin': 'libmkl_rt.dylib',
        'win32': 'mkl_rt.dll',
        'linux': 'libmkl_rt.so',
        'linux2': 'libmkl_rt.so',
    }[sys.platform]

    if sys.platform in ['darwin', 'linux2', 'linux']:
        locations = [
            'lib',
            os.path.join('ext', 'lib'),
        ]
    else:
        locations = [
            os.path.join('Library', 'bin'),
            os.path.join('ext', 'lib'),
        ]

    for location in locations:
        try:
            qset.mkl_lib = cdll.LoadLibrary(
                os.path.join(python_dir, location, library)
            )
            qset.has_mkl = True
            return
        except Exception:
            pass

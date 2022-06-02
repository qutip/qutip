"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
import os, sys
from .utilities import _blas_info
from ctypes import cdll
import platform

__all__ = ['settings']

def _find_mkl():
    """
    Finds the MKL runtime library for the Anaconda and Intel Python
    distributions.
    """
    has_mkl = False
    mkl_lib = None
    if _blas_info() == 'INTEL MKL':
        plat = sys.platform
        python_dir = os.path.dirname(sys.executable)
        if plat in ['darwin','linux2', 'linux']:
            python_dir = os.path.dirname(python_dir)

        if plat == 'darwin':
            lib = '/libmkl_rt.dylib'
        elif plat == 'win32':
            lib = r'\mkl_rt.dll'
        elif plat in ['linux2', 'linux']:
            lib = '/libmkl_rt.so'
        else:
            raise Exception('Unknown platfrom.')

        if plat in ['darwin','linux2', 'linux']:
            lib_dir = '/lib'
        else:
            lib_dir = r'\Library\bin'
        # Try in default Anaconda location first
        try:
            mkl_lib = cdll.LoadLibrary(python_dir+lib_dir+lib)
            has_mkl = True
        except:
            pass

        # Look in Intel Python distro location
        if not has_mkl:
            if plat in ['darwin','linux2', 'linux']:
                lib_dir = '/ext/lib'
            else:
                lib_dir = r'\ext\lib'
            try:
                mkl_lib = \
                    cdll.LoadLibrary(python_dir+lib_dir+lib)
                has_mkl = True
            except:
                pass
    return has_mkl, mkl_lib


class Settings:
    """
    Qutip default settings and options.
    `print(qutip.settings)` to list all available options.
    """
    def __new__(cls):
        """Set Settings as a singleton."""
        if not hasattr(cls, 'instance'):
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._has_mkl, self._mkl_lib = _find_mkl()
        try:
            self.tmproot = os.path.join(os.path.expanduser("~"), '.qutip')
        except OSError:
            self._tmproot = "."
        self.core = None
        self.compilation = None
        self._solvers = []
        self._integrators = []

    @property
    def has_mkl(self):
        return self._has_mkl

    @property
    def mkl_lib(self):
        return self._mkl_lib

    @property
    def ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    @property
    def eigh_unsafe(self):
        return _blas_info() == "OPENBLAS" and platform.system() == 'Darwin'

    @property
    def tmproot(self):
        return self._tmproot

    @tmproot.setter
    def tmproot(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        self._tmproot = root

    @property
    def coeffroot(self):
        return self._coeffroot

    @coeffroot.setter
    def coeffroot(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        self._coeffroot = root

    @property
    def coeff_write_ok(self):
        return os.access(self.coeffroot, os.W_OK)

    @property
    def has_openmp(self):
        return False
        # We keep this as a reminder for when openmp is restored: see Pull #652
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    @property
    def idxint_size(self):
        from .core import data
        return data.base.idxint_size

    @property
    def num_cpus(self):
        from qutip.utilities import available_cpu_count
        if 'QUTIP_NUM_PROCESSES' in os.environ:
            num_cpus = int(os.environ['QUTIP_NUM_PROCESSES'])
        else:
            num_cpus = available_cpu_count()
            os.environ['QUTIP_NUM_PROCESSES'] = str(num_cpus)
        return num_cpus

    def __str__(self):
        lines = []
        for attr in self.__dir__():
            if not attr.startswith('_'):
                lines.append(f"{attr}: {self.__getattribute__(attr)}")
        return '\n'.join(lines)

    def __repr__(self):
        return self.__str__()


settings = Settings()

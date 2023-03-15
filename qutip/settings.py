"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
import os
import sys
from ctypes import cdll
import platform
import numpy as np

__all__ = ['settings']


def _blas_info():
    config = np.__config__
    if hasattr(config, 'blas_ilp64_opt_info'):
        blas_info = config.blas_ilp64_opt_info
    elif hasattr(config, 'blas_opt_info'):
        blas_info = config.blas_opt_info
    else:
        blas_info = {}

    def _in_libaries(name):
        return any(name in lib for lib in blas_info.get('libraries', []))

    if getattr(config, 'mkl_info', False) or _in_libaries("mkl"):
        blas = 'INTEL MKL'
    elif getattr(config, 'openblas_info', False) or _in_libaries('openblas'):
        blas = 'OPENBLAS'
    elif '-Wl,Accelerate' in blas_info.get('extra_link_args', []):
        blas = 'Accelerate'
    else:
        blas = 'Generic'
    return blas


def available_cpu_count():
    """
    Get the number of cpus.
    It tries to only get the number available to qutip.
    """
    import os
    import multiprocessing
    try:
        import psutil
    except ImportError:
        psutil = None
    num_cpu = 0

    if 'QUTIP_NUM_PROCESSES' in os.environ:
        # We consider QUTIP_NUM_PROCESSES=0 as unset.
        num_cpu = int(os.environ['QUTIP_NUM_PROCESSES'])

    if num_cpu == 0 and 'SLURM_CPUS_PER_TASK' in os.environ:
        num_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])

    if num_cpu == 0 and hasattr(os, 'sched_getaffinity'):
        num_cpu = len(os.sched_getaffinity(0))

    if (
        num_cpu == 0
        and psutil is not None
        and hasattr(psutil.Process(), "cpu_affinity")
    ):
        num_cpu = len(psutil.Process().cpu_affinity())

    if num_cpu == 0:
        try:
            num_cpu = multiprocessing.cpu_count()
        except NotImplementedError:
            pass

    return num_cpu or 1


def _find_mkl():
    """
    Finds the MKL runtime library for the Anaconda and Intel Python
    distributions.
    """
    mkl_lib = None
    if _blas_info() == 'INTEL MKL':
        plat = sys.platform
        python_dir = os.path.dirname(sys.executable)
        if plat in ['darwin', 'linux2', 'linux']:
            python_dir = os.path.dirname(python_dir)

        if plat == 'darwin':
            lib = '/libmkl_rt.dylib'
        elif plat == 'win32':
            lib = r'\mkl_rt.dll'
        elif plat in ['linux2', 'linux']:
            lib = '/libmkl_rt.so'
        else:
            raise Exception('Unknown platfrom.')

        if plat in ['darwin', 'linux2', 'linux']:
            lib_dir = '/lib'
        else:
            lib_dir = r'\Library\bin'
        # Try in default Anaconda location first
        try:
            mkl_lib = cdll.LoadLibrary(python_dir+lib_dir+lib)
        except Exception:
            pass

        # Look in Intel Python distro location
        if mkl_lib is None:
            if plat in ['darwin', 'linux2', 'linux']:
                lib_dir = '/ext/lib'
            else:
                lib_dir = r'\ext\lib'
            try:
                mkl_lib = \
                    cdll.LoadLibrary(python_dir + lib_dir + lib)
            except Exception:
                pass
    return mkl_lib


class Settings:
    """
    Qutip's settings and options.
    """
    def __init__(self):
        self._mkl_lib = ""
        try:
            self.tmproot = os.path.join(os.path.expanduser("~"), '.qutip')
        except OSError:
            self._tmproot = "."
        self.core = None  # set in qutip.core.options
        self.compile = None  # set in qutip.core.coefficient
        self._debug = False
        self._log_handler = "default"
        self._colorblind_safe = False

    @property
    def has_mkl(self):
        """ Whether qutip found an mkl installation. """
        return self.mkl_lib is not None

    @property
    def mkl_lib(self):
        """ Location of the mkl installation. """
        if self._mkl_lib == "":
            self._mkl_lib = _find_mkl()
        return _find_mkl()

    @property
    def ipython(self):
        """ Whether qutip is running in ipython. """
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    @property
    def eigh_unsafe(self):
        """
        Whether `eigh` call is reliable.
        Some implementation of blas have some issues on some OS.
        """
        from packaging import version as pac_version
        import scipy
        is_old_scipy = (
            pac_version.parse(scipy.__version__) < pac_version.parse("1.5")
        )
        return (
            # macOS OpenBLAS eigh is unstable, see #1288
            (_blas_info() == "OPENBLAS" and platform.system() == 'Darwin')
            # The combination of scipy<1.5 and MKL causes wrong results when
            # calling eigh for big matrices.  See #1495, #1491 and #1498.
            or (is_old_scipy and (_blas_info() == 'INTEL MKL'))
        )

    @property
    def tmproot(self):
        """
        Location in which qutip place cython string coefficient folders.
        The default is "$HOME/.qutip".
        Can be updated.
        """
        return self._tmproot

    @tmproot.setter
    def tmproot(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        self._tmproot = root

    @property
    def coeffroot(self):
        """
        Location in which qutip save cython string coefficient files.
        Usually "{qutip.settings.tmproot}/qutip_coeffs_X.X".
        Can be updated.
        """
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
        """ Whether qutip has write acces to ``qutip.settings.coeffroot``."""
        return os.access(self.coeffroot, os.W_OK)

    @property
    def has_openmp(self):
        return False
        # We keep this as a reminder for when openmp is restored: see Pull #652
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    @property
    def idxint_size(self):
        """
        Integer type used by ``CSR`` data.
        Sparse ``CSR`` matrices can contain at most ``2**idxint_size``
        non-zeros elements.
        """
        from .core import data
        return data.base.idxint_size

    @property
    def num_cpus(self):
        """
        Number of cpu detected.
        Use the solver options to control the number of cpus used.
        """
        if 'QUTIP_NUM_PROCESSES' in os.environ:
            num_cpus = int(os.environ['QUTIP_NUM_PROCESSES'])
        else:
            num_cpus = available_cpu_count()
            os.environ['QUTIP_NUM_PROCESSES'] = str(num_cpus)
        return num_cpus

    @property
    def debug(self):
        """
        Debug mode for development.
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def log_handler(self):
        """
        Define whether log handler should be:
        - default: switch based on IPython detection
        - stream: set up non-propagating StreamHandler
        - basic: call basicConfig
        - null: leave logging to the user
        """
        return self._log_handler

    @log_handler.setter
    def log_handler(self, value):
        self._log_handler = value

    @property
    def colorblind_safe(self):
        """
        Allow for a colorblind mode that uses different colormaps
        and plotting options by default.
        """
        return self._colorblind_safe

    @colorblind_safe.setter
    def colorblind_safe(self, value):
        self._colorblind_safe = value

    def __str__(self):
        lines = ["Qutip settings:"]
        for attr in self.__dir__():
            if not attr.startswith('_') and attr not in ["core", "compile"]:
                lines.append(f"    {attr}: {self.__getattribute__(attr)}")
        lines.append(f"    compile: {self.compile.__repr__(full=False)}")
        return '\n'.join(lines)

    def __repr__(self):
        return self.__str__()


settings = Settings()

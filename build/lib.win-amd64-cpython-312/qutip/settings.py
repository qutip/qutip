"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
import os
import sys
from ctypes import cdll, CDLL
import platform
from glob import glob
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


def available_cpu_count() -> int:
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
    Finds the MKL library for the Anaconda and Intel Python
    distributions.
    """
    plat = sys.platform

    if plat.startswith("win"):
        # TODO: fix the mkl handling on windows or use modules like pydiso to
        # do it for us.
        return ""
    if plat == "emscripten":
        # mkl is not supported on emscripten
        return ""

    python_dir = os.path.dirname(sys.executable)
    if plat in ['darwin', 'linux2', 'linux']:
        python_dir = os.path.dirname(python_dir)

    if plat == 'darwin':
        ext = ".dylib"
    elif plat == 'win32':
        ext = ".dll"
    elif plat in ['linux2', 'linux']:
        ext = ".so"
    else:
        raise Exception('Unknown platfrom.')

    # Try in default Anaconda location first
    if plat in ['darwin', 'linux2', 'linux']:
        lib_dir = '/lib/*'
    else:
        lib_dir = r'\Library\bin\*'

    libraries = glob(python_dir + lib_dir)
    mkl_libs = [lib for lib in libraries if "mkl_rt" in lib]

    if not mkl_libs:
        # Look in Intel Python distro location
        if plat in ['darwin', 'linux2', 'linux']:
            lib_dir = '/ext/lib'
        else:
            lib_dir = r'\ext\lib'
        libraries = glob(python_dir + lib_dir)
        mkl_libs = [
            lib for lib in libraries
            if "mkl_rt." in lib and ext in lib
        ]

    if mkl_libs:
        # If multiple libs are found, they should all be the same.
        return mkl_libs[-1]
    return ""


class Settings:
    """
    Qutip's settings and options.
    """
    def __init__(self):
        self._mkl_lib = ""
        self._mkl_lib_loc = ""
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
    def has_mkl(self) -> bool:
        """ Whether qutip found an mkl installation. """
        return self.mkl_lib is not None

    @property
    def mkl_lib_location(self) -> str | None:
        """ Location of the mkl library file. The file is usually called:

        - `libmkl_rt.so` (Linux)
        - `libmkl_rt.dylib` (Mac)
        - `mkl_rt.dll` (Windows)

        It search for the library in the python lib path per default.
        If the library is in other location, update this variable as needed.
        """
        if self._mkl_lib_loc == "":
            _mkl_lib_loc = _find_mkl()
            try:
                _mkl_lib = cdll.LoadLibrary(_mkl_lib_loc)
            except OSError:
                _mkl_lib = None
            if not (
                hasattr(_mkl_lib, "pardiso")
                and hasattr(_mkl_lib, "mkl_cspblas_zcsrgemv")
            ):
                self._mkl_lib_loc = None
                self._mkl_lib = None
            else:
                self._mkl_lib = _mkl_lib
                self._mkl_lib_loc = _mkl_lib_loc
        return self._mkl_lib_loc

    @mkl_lib_location.setter
    def mkl_lib_location(self, new: str):
        _mkl_lib = cdll.LoadLibrary(new)
        if not (
            hasattr(_mkl_lib, "pardiso")
            and hasattr(_mkl_lib, "mkl_cspblas_zcsrgemv")
        ):
            raise ValueError(
                "mkl sparse functions not available in the provided library"
            )
        self._mkl_lib_loc = new
        self._mkl_lib = _mkl_lib

    @property
    def mkl_lib(self) -> CDLL | None:
        """ Mkl library """
        if self._mkl_lib == "":
            self.mkl_lib_location
        return self._mkl_lib

    @property
    def ipython(self) -> bool:
        """ Whether qutip is running in ipython. """
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    @property
    def eigh_unsafe(self) -> bool:
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
    def tmproot(self) -> str:
        """
        Location in which qutip place cython string coefficient folders.
        The default is "$HOME/.qutip".
        Can be updated.
        """
        return self._tmproot

    @tmproot.setter
    def tmproot(self, root: str) -> None:
        if not os.path.exists(root):
            os.mkdir(root)
        self._tmproot = root

    @property
    def coeffroot(self) -> str:
        """
        Location in which qutip save cython string coefficient files.
        Usually "{qutip.settings.tmproot}/qutip_coeffs_X.X".
        Can be updated.
        """
        return self._coeffroot

    @coeffroot.setter
    def coeffroot(self, root: str) -> None:
        if not os.path.exists(root):
            os.mkdir(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        self._coeffroot = root

    @property
    def coeff_write_ok(self) -> bool:
        """ Whether qutip has write acces to ``qutip.settings.coeffroot``."""
        return os.access(self.coeffroot, os.W_OK)

    @property
    def _has_openmp(self) -> bool:
        return False
        # We keep this as a reminder for when openmp is restored: see Pull #652
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    @property
    def idxint_size(self) -> int:
        """
        Integer type used by ``CSR`` data.
        Sparse ``CSR`` matrices can contain at most ``2**idxint_size``
        non-zeros elements.
        """
        from .core import data
        return data.base.idxint_size

    @property
    def num_cpus(self) -> int:
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
    def colorblind_safe(self) -> bool:
        """
        Allow for a colorblind mode that uses different colormaps
        and plotting options by default.
        """
        return self._colorblind_safe

    @colorblind_safe.setter
    def colorblind_safe(self, value: bool) -> None:
        self._colorblind_safe = value

    def __str__(self) -> str:
        lines = ["Qutip settings:"]
        for attr in self.__dir__():
            if not attr.startswith('_') and attr not in ["core", "compile"]:
                lines.append(f"    {attr}: {self.__getattribute__(attr)}")
        lines.append(f"    compile: {self.compile.__repr__(full=False)}")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self.__str__()


settings = Settings()

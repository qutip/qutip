"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
import os
import sys
from ctypes import cdll, CDLL
import platform
from glob import glob
from pathlib import Path
import warnings
import numpy as np
import scipy

__all__ = ['settings']

def _blas_info_pre_1_26():
    config = scipy.__config__
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


def _blas_info():
    """
    Find scipy blas and lapack info.
    scipy blas version can be different from numpy's one.
    In cython we link to blas using scipy's cython binding and it's the one
    used for advanced use (eigen, ode, solving linear equation...)
    Therefore it's the relevent one for our usage.
    """
    try:
        config = scipy.show_config("dicts")
    except TypeError:
        return _blas_info_pre_1_26()

    try:
        return config["Build Dependencies"]["blas"]["name"]
    except KeyError:
        return 'Generic'


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


@functools.cache
def _has_pydiso() -> bool:
    try:
        import pydiso.mkl_solver
    except ImportError:
        return False
    return True

class Settings:
    """
    Qutip's settings and options.
    """
    def __init__(self):
        self._mkl_lib = ""
        self._mkl_lib_loc = ""
        try:
            self.tmproot = str(Path.home() / '.qutip')
        except OSError:
            self._tmproot = "."
        self.core = None  # set in qutip.core.options
        self.compile = None  # set in qutip.core.coefficient
        self._debug = False
        self._log_handler = "default"
        self._colorblind_safe = False

    @property
    def has_mkl(self) -> bool:
        """ Checks whether the MKL Pardiso sparse solver is available.
            Requires the optional ``pydiso`` package. """
        return _has_pydiso()

    @property
    def mkl_lib_location(self) -> str | None:
        """ Location of the mkl library file. The file is usually called:

        - `libmkl_rt.so` (Linux)
        - `libmkl_rt.dylib` (Mac)
        - `mkl_rt.dll` (Windows)

        It search for the library in the python lib path per default.
        If the library is in other location, update this variable as needed.
        """
        warnings.warn(
            "The 'mkl_lib_location' property is deprecated; use 'has_mkl' instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return ""

    @mkl_lib_location.setter
    def mkl_lib_location(self, new: str):
        warnings.warn(
            "The 'mkl_lib_location' setter is deprecated; it is not possible to point
            qutip at a libmkl_rt in a non-standard location since pydiso links it at
            a build time.",
            category=DeprecationWarning,
            stacklevel=2
        )

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
        root = Path(root)
        root.mkdir(exist_ok=True)
        self._tmproot = str(root)

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
        root = Path(root)
        root.mkdir(exist_ok=True)
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        self._coeffroot = root_str

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

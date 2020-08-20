from .optionsclass import optionsclass
import sys
import os
import logging
import platform
from .utilities import _blas_info

@optionsclass("install")
class InstallSettings:
    """
    QuTiP's settings

    Options
    -------
    debug: False
        debug mode for development

    log_handler: str
        define whether log handler should be
        - default: switch based on IPython detection
        - stream: set up non-propagating StreamHandler
        - basic: call basicConfig
        - null: leave logging to the user

    colorblind_safe: False
        Allow for a colorblind mode that uses different colormaps
        and plotting options by default.

    tmproot: str
        Location of the saved string coefficients
        Make sure it is in the "sys.path" if changing.

    _logger: Logger
        Logger
        *readonly*

    eigh_unsafe: bool
        Running on mac with openblas make eigh unsafe
        *readonly*

    mkl_lib: str
        location of the mkl library
        *readonly*

    has_mkl: bool
        Flag if mkl_lib is found
        *readonly*

    ipython: bool
        Are we in IPython?
        *readonly*

    """
    try:
        _logger = logging.getLogger(__name__)
        _logger.addHandler(logging.NullHandler())
    except:
        _logger = None

    try:
        qutip_conf_dir = os.path.join(os.path.expanduser("~"), '.qutip')
        if not os.path.exists(qutip_conf_dir):
            os.mkdir(qutip_conf_dir)
        tmproot = os.path.join(qutip_conf_dir, 'coeffs')
        if not os.path.exists(tmproot):
            os.mkdir(tmproot)
        assert os.access(tmproot, os.W_OK)
        del qutip_conf_dir
    except Exception:
        tmproot = "."
    if tmproot not in sys.path:
        sys.path.insert(0, tmproot)

    # ------------------------------------------------------------------------
    # Check if we're in IPython.
    try:
        __IPYTHON__
        _ipython = True
    except NameError:
        _ipython = False

    _eigh_unsafe = (_blas_info() == "OPENBLAS"
                    and platform.system() == 'Darwin')

    options = {
        # debug mode for development
        "debug": False,
        # define whether log handler should be
        #   - default: switch based on IPython detection
        #   - stream: set up non-propagating StreamHandler
        #   - basic: call basicConfig
        #   - null: leave logging to the user
        "log_handler": 'default',
        # Allow for a colorblind mode that uses different colormaps
        # and plotting options by default.
        "colorblind_safe": False,
        # Location of the saved string coefficients
        # Make sure it is in the "sys.path" if changing.
        "tmproot": tmproot
    }

    read_only_options = {
        # location of the mkl library
        "mkl_lib": None,
        # Flag if mkl_lib is found
        "has_mkl": False,
        # are we in IPython?
        "ipython": _ipython,
        # Note that since logging depends on settings,
        # if we want to do any logging here, it must be manually
        # configured, rather than through _logging.get_logger().
        "_logger": _logger,
        # Running on mac with openblas make eigh unsafe
        "eigh_unsafe": _eigh_unsafe
    }
